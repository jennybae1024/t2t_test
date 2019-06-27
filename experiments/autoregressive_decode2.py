# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, re
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import registry
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir
from tensor2tensor.utils import flags
from tensor2tensor import problems
import tensorflow as tf
tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys
flags = tf.flags
FLAGS = flags.FLAGS
# See utils/flags.py for additional command-line flags.
flags.DEFINE_integer("test_shard", -1, "test shard -1: all.")
flags.DEFINE_string("global_steps", "", "global_steps.")
flags.DEFINE_string("checkpoint_path", "", "model directory.")
flags.DEFINE_string("my_data_dir", "/media/disk1/ksk5693/data/t2t_data/", "Data directory.")




flags.DEFINE_string("t2t_usr_dir", None,
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_model calls, that will then be "
                    "available to the t2t-trainer.")
flags.DEFINE_integer("random_seed", None, "Random seed.")
flags.DEFINE_integer("tpu_num_shards", 8, "Number of tpu shards.")
flags.DEFINE_integer("iterations_per_loop", 100,
                     "Number of iterations in a TPU training loop.")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU.")
flags.DEFINE_bool("use_tpu_estimator", False, "Whether to use TPUEstimator. "
                                              "This is always enabled when use_tpu is True.")
flags.DEFINE_bool("xla_compile", False,
                  "Whether to use XLA to compile model_fn.")
flags.DEFINE_integer("xla_jit_level", -1,
                     "GlobalJitLevel to use while compiling the full graph.")
flags.DEFINE_integer("tpu_infeed_sleep_secs", None,
                     "How long to sleep the infeed thread.")
flags.DEFINE_bool("generate_data", False, "Generate data before training?")
flags.DEFINE_string("tmp_dir", "/tmp/t2t_datagen",
                    "Temporary storage directory, used if --generate_data.")
flags.DEFINE_bool("profile", False, "Profile performance?")
flags.DEFINE_integer("inter_op_parallelism_threads", 0,
                     "Number of inter_op_parallelism_threads to use for CPU. "
                     "See TensorFlow config.proto for details.")
flags.DEFINE_integer("intra_op_parallelism_threads", 0,
                     "Number of intra_op_parallelism_threads to use for CPU. "
                     "See TensorFlow config.proto for details.")
# TODO(lukaszkaiser): resolve memory and variable assign issues and set to True.
flags.DEFINE_bool(
    "optionally_use_dist_strat", False,
    "Whether to use TensorFlow DistributionStrategy instead of explicitly "
    "replicating the model. DistributionStrategy is used only if the "
    "model replication configuration is supported by the DistributionStrategy.")
# To maintain compatibility with some internal libs, we guard against these flag
# definitions possibly erroring. Apologies for the ugliness.
try:
    flags.DEFINE_string("master", "", "Address of TensorFlow master.")
    flags.DEFINE_string("output_dir", "", "Base output directory for run.")
    flags.DEFINE_string("schedule", "continuous_train_and_eval",
                        "Method of Experiment to run.")
    flags.DEFINE_integer("eval_steps", 100,
                         "Number of steps in evaluation. By default, eval will "
                         "stop after eval_steps or when it runs through the eval "
                         "dataset once in full, whichever comes first, so this "
                         "can be a very large number.")
except:  # pylint: disable=bare-except
    pass

flags.DEFINE_string("std_server_protocol", "grpc",
                    "Protocol for tf.train.Server.")

# Google Cloud TPUs
flags.DEFINE_string("cloud_tpu_name", "%s-tpu" % os.getenv("USER"),
                    "Name of Cloud TPU instance to use or create.")

# Google Cloud ML Engine
flags.DEFINE_bool("cloud_mlengine", False,
                  "Whether to launch on Cloud ML Engine.")
flags.DEFINE_string("cloud_mlengine_master_type", None,
                    "Machine type for master on Cloud ML Engine. "
                    "If provided, overrides default selections based on "
                    "--worker_gpu. User is responsible for ensuring "
                    "type is valid and that --worker_gpu matches number of "
                    "GPUs on machine type. See documentation: "
                    "https://cloud.google.com/ml-engine/reference/rest/v1/"
                    "projects.jobs#traininginput")
# Hyperparameter tuning on Cloud ML Engine
# Pass an --hparams_range to enable
flags.DEFINE_string("autotune_objective", None,
                    "TensorBoard metric name to optimize.")
flags.DEFINE_bool("autotune_maximize", True,
                  "Whether to maximize (vs. minimize) autotune_objective.")
flags.DEFINE_integer("autotune_max_trials", 10,
                     "Maximum number of tuning experiments to run.")
flags.DEFINE_integer("autotune_parallel_trials", 1,
                     "How many trials to run in parallel (will spin up this "
                     "many jobs.")
# Note than in open-source TensorFlow, the dash gets converted to an underscore,
# so access is FLAGS.job_dir.
flags.DEFINE_string("job-dir", None,
                    "DO NOT USE. Exists only for Cloud ML Engine to pass in "
                    "during hyperparameter tuning. Overrides --output_dir.")
flags.DEFINE_integer("log_step_count_steps", 100,
                     "Number of local steps after which progress is printed "
                     "out")


# Additional flags in bin/t2t_trainer.py and utils/flags.py
flags.DEFINE_bool("keep_timestamp", False,
                  "Set the mtime of the decoded file to the "
                  "checkpoint_path+'.index' mtime.")
flags.DEFINE_bool("decode_interactive", False,
                  "Interactive local inference mode.")
flags.DEFINE_integer("decode_shards", 1, "Number of decoding replicas.")
flags.DEFINE_string("score_file", "", "File to score. Each line in the file "
                                      "must be in the format input \t target.")
flags.DEFINE_bool("decode_in_memory", False, "Decode in memory.")






flags.DEFINE_integer("gpu_num", 0, "gpu_num")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_num)

def create_hparams():
    return trainer_lib.create_hparams(
        FLAGS.hparams_set,
        FLAGS.hparams,
        data_dir=os.path.expanduser(FLAGS.my_data_dir),
        problem_name=FLAGS.problem)


def create_decode_hparams(shard):
    decode_hp = decoding.decode_hparams("beam_size=1")
    decode_hp.shards = FLAGS.decode_shards
    decode_hp.shard_id = shard
    decode_in_memory = FLAGS.decode_in_memory or decode_hp.decode_in_memory
    decode_hp.decode_in_memory = decode_in_memory
    if FLAGS.global_steps:
        decode_hp.decode_to_file = str(FLAGS.global_steps)+"test"+str(shard)
    else:
        decode_hp.decode_to_file = "test"+str(shard)
    decode_hp.decode_reference = FLAGS.decode_reference
    decode_hp.log_results = True
    decode_hp.batch_size = 128
    return decode_hp


def decode(estimator, hparams, decode_hp):
    """Decode from estimator. Interactive, from file, or from dataset."""
    if FLAGS.decode_interactive:
        if estimator.config.use_tpu:
            raise ValueError("TPU can only decode from dataset.")
        decoding.decode_interactively(estimator, hparams, decode_hp,
                                      checkpoint_path=FLAGS.checkpoint_path)
    elif FLAGS.decode_from_file:
        decoding.decode_from_file(estimator, FLAGS.decode_from_file, hparams,
                                  decode_hp, FLAGS.decode_to_file,
                                  checkpoint_path=FLAGS.checkpoint_path)
        if checkpoint_path and FLAGS.keep_timestamp:
            ckpt_time = os.path.getmtime(FLAGS.checkpoint_path + ".index")
            os.utime(FLAGS.decode_to_file, (ckpt_time, ckpt_time))
    else:
        decoding.decode_from_dataset(
            estimator,
            FLAGS.problem,
            hparams,
            decode_hp,
            decode_to_file=FLAGS.decode_to_file,
            dataset_split="test",
            checkpoint_path=FLAGS.checkpoint_path)


def score_file(filename):
    """Score each line in a file and return the scores."""
    # Prepare model.
    hparams = create_hparams()
    encoders = registry.problem(FLAGS.problem).feature_encoders(FLAGS.my_data_dir)
    has_inputs = "inputs" in encoders

    # Prepare features for feeding into the model.
    if has_inputs:
        inputs_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
        batch_inputs = tf.reshape(inputs_ph, [1, -1, 1, 1])  # Make it 4D.
    targets_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
    batch_targets = tf.reshape(targets_ph, [1, -1, 1, 1])  # Make it 4D.
    if has_inputs:
        features = {"inputs": batch_inputs, "targets": batch_targets}
    else:
        features = {"targets": batch_targets}

    # Prepare the model and the graph when model runs on features.
    model = registry.model(FLAGS.model)(hparams, tf.estimator.ModeKeys.EVAL)
    _, losses = model(features)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Load weights from checkpoint.
        saver.restore(sess, FLAGS.checkpoint_path)
        # Run on each line.
        with tf.gfile.Open(filename) as f:
            lines = f.readlines()
        results = []
        for line in lines:
            tab_split = line.split("\t")
            if len(tab_split) > 2:
                raise ValueError("Each line must have at most one tab separator.")
            if len(tab_split) == 1:
                targets = tab_split[0].strip()
            else:
                targets = tab_split[1].strip()
                inputs = tab_split[0].strip()
            # Run encoders and append EOS symbol.
            targets_numpy = encoders["targets"].encode(
                targets) + [text_encoder.EOS_ID]
            if has_inputs:
                inputs_numpy = encoders["inputs"].encode(inputs) + [text_encoder.EOS_ID]
            # Prepare the feed.
            if has_inputs:
                feed = {inputs_ph: inputs_numpy, targets_ph: targets_numpy}
            else:
                feed = {targets_ph: targets_numpy}
            # Get the score.
            np_loss = sess.run(losses["training"], feed)
            results.append(np_loss)
    return results


def create_run_config(hp, output_dir=None):
    """Create a run config.

    Args:
      hp: model hyperparameters
      output_dir: model's output directory, defaults to output_dir flag.

    Returns:
      a run config
    """
    save_ckpt_steps = max(FLAGS.iterations_per_loop, FLAGS.local_eval_frequency)
    save_ckpt_secs = FLAGS.save_checkpoints_secs or None
    if save_ckpt_secs:
        save_ckpt_steps = None
    assert FLAGS.output_dir or FLAGS.checkpoint_path
    tpu_config_extra_kwargs = {}

    if getattr(hp, "mtf_mode", False):
        save_ckpt_steps = None  # Disable the default saver
        save_ckpt_secs = None  # Disable the default saver
        tpu_config_extra_kwargs = {
            "num_cores_per_replica": 1,
            "per_host_input_for_training": tpu_config.InputPipelineConfig.BROADCAST,
        }

    # the various custom getters we have written do not play well together yet.
    # TODO(noam): ask rsepassi for help here.
    daisy_chain_variables = (
            hp.daisy_chain_variables and
            hp.activation_dtype == "float32" and
            hp.weight_dtype == "float32")
    return trainer_lib.create_run_config(
        model_name=FLAGS.model,
        model_dir=output_dir or os.path.expanduser(FLAGS.output_dir),
        master=FLAGS.master,
        iterations_per_loop=FLAGS.iterations_per_loop,
        num_shards=FLAGS.tpu_num_shards,
        log_device_placement=FLAGS.log_device_placement,
        save_checkpoints_steps=save_ckpt_steps,
        save_checkpoints_secs=save_ckpt_secs,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours,
        num_gpus=FLAGS.worker_gpu,
        gpu_order=FLAGS.gpu_order,
        num_async_replicas=FLAGS.worker_replicas,
        gpu_mem_fraction=FLAGS.worker_gpu_memory_fraction,
        enable_graph_rewriter=FLAGS.enable_graph_rewriter,
        use_tpu=FLAGS.use_tpu,
        use_tpu_estimator=FLAGS.use_tpu_estimator,
        xla_jit_level=FLAGS.xla_jit_level,
        schedule=FLAGS.schedule,
        no_data_parallelism=hp.no_data_parallelism,
        optionally_use_dist_strat=FLAGS.optionally_use_dist_strat,
        daisy_chain_variables=daisy_chain_variables,
        ps_replicas=FLAGS.ps_replicas,
        ps_job=FLAGS.ps_job,
        ps_gpu=FLAGS.ps_gpu,
        sync=FLAGS.sync,
        worker_id=FLAGS.worker_id,
        worker_job=FLAGS.worker_job,
        random_seed=FLAGS.random_seed,
        tpu_infeed_sleep_secs=FLAGS.tpu_infeed_sleep_secs,
        inter_op_parallelism_threads=FLAGS.inter_op_parallelism_threads,
        log_step_count_steps=FLAGS.log_step_count_steps,
        intra_op_parallelism_threads=FLAGS.intra_op_parallelism_threads,
        tpu_config_extra_kwargs=tpu_config_extra_kwargs,
        cloud_tpu_name=FLAGS.cloud_tpu_name)

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    trainer_lib.set_random_seed(FLAGS.random_seed)
    usr_dir.import_usr_dir("~/t2t/usr_dir")


    if FLAGS.score_file:
        filename = os.path.expanduser(FLAGS.score_file)
        if not tf.gfile.Exists(filename):
            raise ValueError("The file to score doesn't exist: %s" % filename)
        results = score_file(filename)
        if not FLAGS.decode_to_file:
            raise ValueError("To score a file, specify --decode_to_file for results.")
        write_file = tf.gfile.Open(os.path.expanduser(FLAGS.decode_to_file), "w")
        for score in results:
            write_file.write("%.6f\n" % score)
        write_file.close()
        return

    hp = create_hparams()

    problem_name = FLAGS.problem
    model = FLAGS.model
    hparams = FLAGS.hparams_set
    output_dir = os.path.expanduser("/media/disk1/ksk5693/sessions/t2t/"+problem_name+'-'+model+'-'+hparams)
    if FLAGS.global_steps:
        FLAGS.checkpoint_path = output_dir+"/model.ckpt-"+str(FLAGS.global_steps)
    else :
        FLAGS.checkpoint_path = tf.train.latest_checkpoint(output_dir)

# Extract Embedding
    translated_model = registry.model(model)(hparams, Modes.EVAL)
    with tfe.restore_variables_on_create(FLAGS.checkpoint_path):
        test = translated_model.extract_embedding()
    new_emb = test['emb'][0]
    new_emb = tf.concat([new_emb[:110], -1.0*new_emb[110:119], new_emb[119:]], axis = 0)
    if FLAGS.test_shard == -1:
        total_data_names = [name for name in os.listdir(FLAGS.my_data_dir) if len(re.findall(problem_name+"-", name)) > 0]
        num = len([name for name in total_data_names if len(re.findall("test", name)) > 0])
        test_shards_list = [i for i in range(num)]
    if FLAGS.test_shard == -2:
        test_shards_list = [0,1,2]
    else:
        test_shards_list = [FLAGS.test_shard]
    for shard in test_shards_list:
        tf.reset_default_graph()
        decode_hp = create_decode_hparams(shard)
        estimator = trainer_lib.create_estimator(
            FLAGS.model,
            hp,
            create_run_config(hp, output_dir),
            decode_hparams=decode_hp,
            use_tpu=FLAGS.use_tpu)
        decode(estimator, hp, decode_hp)
        print("shard "+str(shard)+" completed")

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
