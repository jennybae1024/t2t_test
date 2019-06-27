from tensor2tensor.utils import usr_dir
from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.utils import hparams_lib, trainer_lib, t2t_model
from tensor2tensor.utils import registry
from datetime import datetime
import tensorflow as tf
import os
tfe = tf.contrib.eager
tfe.enable_eager_execution()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("hparams", "",
                    "A comma-separated list of `name=value` hyperparameter "
                    "values. This flag is used to override hyperparameter "
                    "settings either when manually selecting hyperparameters "
                    "or when using Vizier. If a hyperparameter setting is "
                    "specified by this flag then it must be a valid "
                    "hyperparameter name for the model.")

flags.DEFINE_string("problem", "", "The name of the problem to solve.")
# tiling_interpolation
# tiling_sanity
# tiling_extension
# algorithmic_math_two_variables_ext

flags.DEFINE_string("model_name", "", "model_name.")
# lstm_seq2seq_attention_bidirectional_encoder
# transformer
# rm_seq2seq
# universal_transformer

flags.DEFINE_string("hparam_set", "", "hyperparameter set.")
# lstm_seq2seq
# transformer_base_single_gpu
# rm_base

flags.DEFINE_string("train_steps", "15000", "total_train_steps")
flags.DEFINE_string("save_steps", "4000", "Save_ckpt_every")
flags.DEFINE_integer("fresh", 0, "0: continue, 1: from the scratch")
flags.DEFINE_integer("gpu_num", 0, "gpu_num")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_num)


flags.DEFINE_string("data_dir", "/media/disk1/jennybae/data/t2t_data/", "Data directory.")
flags.DEFINE_string("tmp_dir", "/home/jennybae/t2t_systematic_generalization/t2t/tmp", "Temporary storage directory.")

flags.DEFINE_string("t2t_usr_dir", "/home/jennybae/t2t_systematic_generalization/usr_dir/",
                    "Path to a Python module that will be imported. The "
                    "__init__.py file should include the necessary imports. "
                    "The imported files should contain registrations, "
                    "e.g. @registry.register_problem calls, that will then be "
                    "available to t2t-datagen.")

problems.problem('image_ms_co_co')
def main(_):
    # Other setup
    Modes = tf.estimator.ModeKeys
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)

    # ### 1. Set directories
    problem_name = FLAGS.problem
    model_name = FLAGS.model_name
    hparam_set = FLAGS.hparam_set
    data_dir = os.path.expanduser(FLAGS.data_dir)
    tmp_dir = os.path.expanduser(FLAGS.tmp_dir)
    train_dir = os.path.expanduser("/media/disk1/jennybae/sessions/t2t/"+problem_name+'-'+model_name+'-'+hparam_set)
    tf.gfile.MakeDirs(tmp_dir)
    tf.gfile.MakeDirs(train_dir)

    # hparams_override = None
    # hparams_override = 'batch_size=4096, clip_grad_norm=10'
    # hparams_override = 'learning_rate=0.01'

    hparams = trainer_lib.create_hparams(hparam_set, FLAGS.hparams,
                                         )
    # hparams.add_hparam("attention_mechanism", "luong")
    # hparams.num_heads = 4
    # hparams.add_hparam("output_attention", 1)
    # hparams.add_hparam("attention_layer_size", 128)
    hparams.data_dir = FLAGS.data_dir
    hparams_lib.add_problem_hparams(hparams, problem_name)

    problem = hparams.problem
    iterations_per_loop = 100
    train_steps = int(FLAGS.train_steps)
    eval_steps = 200
    schedule = "train_and_evaluate"
    warm_start_from = None
    std_server_protocol = None
    min_eval_frequency = int(FLAGS.save_steps)
    eval_timeout_mins = 240
    run_config = trainer_lib.create_run_config(model_name=model_name,
                                               model_dir=train_dir,
                                               iterations_per_loop=iterations_per_loop,
                                               no_data_parallelism=hparams.no_data_parallelism,
                                               keep_checkpoint_max=200,
                                               save_checkpoints_steps=int(FLAGS.save_steps),
                                               )


    hparams.add_hparam("model_dir", run_config.model_dir)
    hparams.add_hparam("train_steps", train_steps)
    hparams.add_hparam("eval_steps", eval_steps)
    hparams.add_hparam("schedule", schedule)
    hparams.add_hparam("warm_start_from", warm_start_from)
    hparams.add_hparam("std_server_protocol", std_server_protocol)
    hparams.add_hparam("eval_freq_in_steps", min_eval_frequency)
    hparams.add_hparam("eval_timeout_mins", eval_timeout_mins)
    hparams_lib.add_problem_hparams(hparams, problem_name)



    train_input_fn = problem.make_estimator_input_fn(mode = tf.estimator.ModeKeys.TRAIN, \
                                                     hparams = hparams, \
                                                     data_dir = data_dir)
    eval_input_fn = problem.make_estimator_input_fn(mode = tf.estimator.ModeKeys.EVAL, \
                                                    hparams = hparams, \
                                                    data_dir = data_dir)


    model_fn = t2t_model.T2TModel.make_estimator_model_fn(model_name, hparams)
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
    )

    # exporter
    exporter = []

    # Hooks
    eval_early_stopping_steps = None
    eval_early_stopping_metric = None
    eval_early_stopping_metric_minimize = True
    eval_early_stopping_metric_delta = None
    validation_monitor_kwargs = dict(
        input_fn=eval_input_fn,
        eval_steps=eval_steps,
        every_n_steps=min_eval_frequency,
        early_stopping_rounds=eval_early_stopping_steps,
        early_stopping_metric=eval_early_stopping_metric,
        early_stopping_metric_minimize=eval_early_stopping_metric_minimize)
    dbgprofile_kwargs = {"output_dir": run_config.model_dir}
    early_stopping_kwargs = dict(
        events_dir=os.path.join(run_config.model_dir, "eval_continuous"),
        tag=eval_early_stopping_metric,
        num_plateau_steps=eval_early_stopping_steps,
        plateau_decrease=eval_early_stopping_metric_minimize,
        plateau_delta=eval_early_stopping_metric_delta,
        every_n_steps=min_eval_frequency)

    use_validation_monitor = (
            schedule == "train_and_evaluate" and min_eval_frequency)
    # Distributed early stopping
    local_schedules = ["train_and_evaluate", "continuous_train_and_eval"]
    use_early_stopping = (
            schedule not in local_schedules and eval_early_stopping_steps)


    use_tfdbg = False
    use_dbgprofile = False
    train_hooks, eval_hooks = trainer_lib.create_hooks(
        use_tfdbg=use_tfdbg,
        use_dbgprofile=use_dbgprofile,
        dbgprofile_kwargs=dbgprofile_kwargs,
        use_validation_monitor=use_validation_monitor,
        validation_monitor_kwargs=validation_monitor_kwargs,
        use_early_stopping=use_early_stopping,
        early_stopping_kwargs=early_stopping_kwargs)

    hook_context = trainer_lib.HookContext(
        estimator=estimator, problem=problem, hparams=hparams)

    train_hooks += t2t_model.T2TModel.get_train_hooks(model_name, hook_context)
    eval_hooks += t2t_model.T2TModel.get_eval_hooks(model_name, hook_context)

    additional_train_hooks = []
    additional_eval_hooks = []
    if additional_train_hooks:
        train_hooks += additional_train_hooks
    if additional_eval_hooks:
        eval_hooks += additional_eval_hooks

    train_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
        train_hooks, estimator)
    eval_hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(
        eval_hooks, estimator)

    train_spec = tf.estimator.TrainSpec(
        train_input_fn, max_steps=train_steps, hooks=train_hooks)

    eval_throttle_seconds = 600
    eval_spec = tf.estimator.EvalSpec(
        eval_input_fn,
        steps=eval_steps,
        hooks=eval_hooks,
        start_delay_secs=0 if hparams.schedule == "evaluate" else 120,
        throttle_secs=eval_throttle_seconds,
        exporters=exporter)

    if FLAGS.fresh == 1:
        tf.gfile.DeleteRecursively(train_dir)
        tf.gfile.MakeDirs(train_dir)

    tf.logging.set_verbosity(tf.logging.INFO)
    time_start = datetime.utcnow()
    print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    print(".......................................")
    tf.estimator.train_and_evaluate(
        estimator=estimator,
        train_spec=train_spec,
        eval_spec=eval_spec
    )

    time_end = datetime.utcnow()
    print(".......................................")
    print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    print("")
    time_elapsed = time_end - time_start
    print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
