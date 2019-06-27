from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.training import session_run_hook
from tensorflow.python.training.session_run_hook import SessionRunArgs
import collections
import tensorflow as tf
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
np.random.seed(0)
def mkdir(path):
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Delete %s " % path)
        shutil.rmtree(path)
        os.mkdir(path)
        print("Remake %s " % path)
    else:  
        print ("Successfully created the directory %s " % path)

# class ExampleHook(session_run_hook.SessionRunHook):
#     def begin(self):
#       # You can add ops to the graph here.
#         print('Starting the session.')
#         TAGS = ['enc_inp', 'predicted_syms', 'labels']
#         def funnel():
#             res = {}
#             for n in tf.get_default_graph().as_graph_def().node:
#                 for tag in TAGS:
#                     if tag in n.name:
#                         res[n.name]=tf.get_default_graph().get_tensor_by_name(n.name+":0")
#             return res
#         self.your_tensor = funnel()
#     def after_create_session(self, session, coord):
#       # When this is called, the graph is finalized and
#       # ops can no longer be added to the graph.
#         print('Session created.')
#     def before_run(self, run_context):
# #         print('Before calling session.run().')
#         return SessionRunArgs(self.your_tensor)
#     def after_run(self, run_context, run_values):
#         for key, values in run_values.results.items():
#             print(key)
#             print(values[0])
#     def end(self, session):
#         print('Done with the session.')    


class EvalResultsExporter(tf.estimator.Exporter):
    """Passed into an EvalSpec for saving the result of the final evaluation
    step locally or in Google Cloud Storage.
    """

    def __init__(self, name):
        assert name, '"name" argument is required.'
        self._name = name

    @property
    def name(self):
        return self._name

    def export(self, estimator, export_path, checkpoint_path,
               eval_result, is_the_final_export):

        if not is_the_final_export:
            return None

        tf.logging.info(('EvalResultsExporter (name: %s) '
                         'running after final evaluation.') % self._name)
        tf.logging.info('export_path: %s' % export_path)
        tf.logging.info('eval_result: %s' % eval_result)

        for key, value in eval_result.items():
            if isinstance(value, np.float32):
                eval_result[key] = value.item()

        tf.gfile.MakeDirs(export_path)
        filehandler = open('%s/eval_results.json' % export_path, "wb")
        pickle.dump(eval_result, filehandler)
        # with tf.gfile.GFile('%s/eval_results.json' % export_path, 'w') as f:
        #     f.write(json.dumps(eval_result))



class ExampleHook(session_run_hook.SessionRunHook):
    def __init__(self, decode):
        self.decode = decode
    def begin(self):
      # You can add ops to the graph here.
        print('Starting the session.')
        TAGS = ['inputs', 'outputs', 'targets']
        res = {}
        for n in tf.get_default_graph().as_graph_def().node:
            for tag in TAGS:
                if tag == n.name[-len(tag):]:
                    res[n.name]=tf.get_default_graph().get_tensor_by_name(n.name+":0")
        self.your_tensor = res
    def after_create_session(self, session, coord):
      # When this is called, the graph is finalized and
      # ops can no longer be added to the graph.
        print('Session created.')
    def before_run(self, run_context):
#         print('Before calling session.run().')
        return SessionRunArgs(self.your_tensor)
    def after_run(self, run_context, run_values):
        for key, values in run_values.results.items():
            print(key)
            print(self.decode(values[0]))
    def end(self, session):
        print('Done with the session.')




class ParameterHook(session_run_hook.SessionRunHook):
    def __init__(self):
        self.step = 0
    def after_run(self, run_context, run_values):
        self.step+=1        
        if self.step>0:
            run_context.request_stop()    
        print('Number of parameters : %d' % sum([np.prod(var.get_shape()) for var in tf.trainable_variables()]))
        
class ParameterHook_write(ParameterHook):
    def __init__(self, output_dir):
        self.step = 0
        self.output_dir = output_dir
    def after_run(self, run_context, run_values):
        self.step+=1
        if self.step>0:
            run_context.request_stop()
        with open(self.output_dir, "a") as myfile:
            myfile.write('Number of parameters : %d \n' % sum([np.prod(var.get_shape()) for var in \
                                                               tf.trainable_variables()]))


class SimpleHook(session_run_hook.SessionRunHook):
    def begin(self):
      # You can add ops to the graph here.
        print('Starting the session.')
        TAGS = ['predicted_syms', 'labels']
        def funnel():
            res = {}
            for n in tf.get_default_graph().as_graph_def().node:
                for tag in TAGS:
                    if tag in n.name:
                        res[n.name]=tf.get_default_graph().get_tensor_by_name(n.name+":0")
            return res
        self.your_tensor = funnel()
    def after_create_session(self, session, coord):
      # When this is called, the graph is finalized and
      # ops can no longer be added to the graph.
        print('Session created.')
    def before_run(self, run_context):
#         print('Before calling session.run().')
        return SessionRunArgs(self.your_tensor)
    def after_run(self, run_context, run_values):
        temp = list(run_values.results.values())
        print(temp[0][0])
        print(temp[0][1])
        print([1 if i==j else 0 for i, j in zip(temp[0][0],temp[0][1])])       
    def end(self, session):
        print('Done with the session.')        
        
        

def Tensor_Hook_generator(tensor_names_list, per_iteration=-1):
    class Hook(session_run_hook.SessionRunHook):
        def __init__(self):
            self.step = 0
        def begin(self):
          # You can add ops to the graph here.
            self.your_tensor = {}
            for n in tf.get_default_graph().as_graph_def().node:
                for tag in tensor_names_list:
                    if tag == n.name[-len(tag):] :
                        self.your_tensor[n.name]=tf.get_default_graph().get_tensor_by_name(n.name+":0")
        def before_run(self, run_context):
    #         print('Before calling session.run().')
            return SessionRunArgs(self.your_tensor)
        def after_run(self, run_context, run_values):            
            if self.step % per_iteration ==0:
                for key in tensor_names_list:
                    print(key)
                    print(run_values.results[key][0])
            self.step += 1
            if per_iteration == -1:
                run_context.request_stop()    
    return Hook()

   
    
    
        
        
class First_logit_Hook(session_run_hook.SessionRunHook):
    def __init__(self):
        self.step = 0
    def begin(self):
      # You can add ops to the graph here.
        TAGS = [ 'eval']
        def funnel():
            res = {}
            for n in tf.get_default_graph().as_graph_def().node:
                for tag in TAGS:
                    if tag in n.name[-len(tag)-1:] :
                        res[n.name]=tf.get_default_graph().get_tensor_by_name(n.name+":0")
            return res
        self.your_tensor = funnel()
    def before_run(self, run_context):
#         print('Before calling session.run().')
        return SessionRunArgs(self.your_tensor)
    def after_run(self, run_context, run_values):
        keys = list(run_values.results.keys())
        keys.sort()
        for key in keys:
            print(key)
            print(run_values.results[key][0][0][:5])
        self.step += 1
        if self.step > 0:
            run_context.request_stop()
    def end(self, session):
        print('Done with the session.')
                
        
class Dec_inp_Hook(session_run_hook.SessionRunHook):
    def __init__(self):
        self.step = 0
    def begin(self):
      # You can add ops to the graph here.
        TAGS = ['dec_inp_']
        def funnel():
            res = {}
            for n in tf.get_default_graph().as_graph_def().node:
                for tag in TAGS:
                    if tag == n.name[-9:-1] :
                        res[n.name]=tf.get_default_graph().get_tensor_by_name(n.name+":0")
            return res
        self.your_tensor = funnel()
    def before_run(self, run_context):
#         print('Before calling session.run().')
        return SessionRunArgs(self.your_tensor)
    def after_run(self, run_context, run_values):
        keys = list(run_values.results.keys())
        keys.sort()
        for key in keys:
            print(key)
            print(run_values.results[key][0])
        self.step += 1
        if self.step > 0:
            run_context.request_stop()
    def end(self, session):
        print('Done with the session.')
        
        
class Train_Hook(session_run_hook.SessionRunHook):
    def __init__(self):
        self.step = 0
    def begin(self):
      # You can add ops to the graph here.
        TAGS = ['predicted_syms', 'labels']
        def funnel():
            res = {}
            for n in tf.get_default_graph().as_graph_def().node:
                for tag in TAGS:
                    if tag == n.name[-len(tag):] :
                        res[n.name]=tf.get_default_graph().get_tensor_by_name(n.name+":0")
            return res
        self.your_tensor = funnel()
    def before_run(self, run_context):
#         print('Before calling session.run().')
        return SessionRunArgs(self.your_tensor)
    def after_run(self, run_context, run_values):
        self.step += 1
        if self.step % 100 == 0:
            keys = list(run_values.results.keys())
            keys.sort()
            for key in keys:
                print(key)
                print(run_values.results[key][0])
    def end(self, session):
        print('Done with the session.')
        
                
        
class AttenHook(session_run_hook.SessionRunHook):
    def __init__(self, params):
        self.params=params
        self.step = 0
    def begin(self):
      # You can add ops to the graph here.
        print('Starting the session.')
        TAGS = ['alphas']
        INPS = ['enc_inp', 'dec_inp']
        self.your_tensor = {}
        for n in tf.get_default_graph().as_graph_def().node:
            for tag in TAGS:
                if tag == n.name[-6:] and '15' in n.name :
                    self.your_tensor[n.name]=tf.get_default_graph().get_tensor_by_name(n.name+":0")
                if tag == n.name[-6:] in n.name :
                    self.your_tensor[n.name]=tf.get_default_graph().get_tensor_by_name(n.name+":0")                    
            for inp in INPS:
                if inp in n.name:
                    self.your_tensor[n.name]=tf.get_default_graph().get_tensor_by_name(n.name+":0")
    def before_run(self, run_context):
#         print('Before calling session.run().')
        return SessionRunArgs(self.your_tensor)
    def after_run(self, run_context, run_values):
        plot_atten(run_values.results, self.params)
        self.step += 1
        if self.step > 0:
            run_context.request_stop()
    def end(self, session):
        print('Done with the session.')

    
    
def plot_atten(atten_dict, params, idx = 0):
    for key, value in atten_dict.items():
        if 'alphas' in key:
            print(key)
            fig, ax1 = plt.subplots(1,1)
            ax1.imshow(value[idx], cmap='hot', interpolation='nearest')        
            if 'enc' in key :
                ax1.set_xticks(np.arange(0, params['in_len'], 1));
                ax1.set_xlabel(atten_dict['enc_inp'][0]);
                ax1.set_yticks(np.arange(0, params['in_len'],  1));
                ax1.set_ylabel(np.array([atten_dict['enc_inp'][0][params['out_len']-1-i] \
                                         for i in range(params['out_len'])]));     
                ax1.set_xticks(np.arange(-.5, float(params['in_len'])+.5, 1), minor=True);
                ax1.set_yticks(np.arange(-.5, float(params['in_len'])+.5, 1), minor=True);
            elif 'self' in key and 'dec' in key :
                ax1.set_xticks(np.arange(0, params['out_len'], 1));
                ax1.set_xlabel(atten_dict['dec_inp'][0]);
                ax1.set_yticks(np.arange(0, params['out_len'],  1));
                ax1.set_ylabel(np.array([atten_dict['dec_inp'][0][params['out_len']-1-i] \
                                         for i in range(params['out_len'])]));      
                ax1.set_xticks(np.arange(-.5, float(params['out_len'])+.5, 1), minor=True);
                ax1.set_yticks(np.arange(-.5, float(params['out_len'])+.5, 1), minor=True);                   
            elif 'int' in key:                
                ax1.set_xticks(np.arange(0, params['in_len'], 1));
                ax1.set_xlabel(atten_dict['enc_inp'][0]);
                ax1.set_yticks(np.arange(0, params['out_len'],  1));
                ax1.set_ylabel(np.array([atten_dict['dec_inp'][0][params['out_len']-1-i] \
                                         for i in range(params['out_len'])]));  
                ax1.set_xticks(np.arange(-.5, float(params['in_len'])+.5, 1), minor=True);
                ax1.set_yticks(np.arange(-.5, float(params['out_len'])+.5, 1), minor=True);
            ax1.grid(which='minor', color='w', linestyle='-', linewidth=0.2)
            plt.show()
            print('---------------------')


def type_show(shard):
    if shard < 6:
        # ran = [-1000, 1000]
        ran = ' -1000 to 1000 '
    else:
        # ran = [-2000, -1000, 1000, 2000]
        ran = ' -2000 to -1000 and 1000 to 2000 '
    if shard % 6 == 0:  # x+x
        # print('x+x', ran)
        return 'x+x' + ran
    if shard % 6 == 1:  # x-x
        # print('x-x', ran)
        return 'x-x' + ran
    if shard % 6 == 2:  # x*x
        # print('x*x', ran)
        return 'x*x' + ran
    if shard % 6 == 3:  # x+y
        # print('x+y', ran)
        return 'x+y' + ran
    if shard % 6 == 4:  # x-y
        # print('x-y', ran)
        return 'x-y' + ran
    if shard % 6 == 5:  # x*y
        # print('x*y', ran)
        return 'x*y' + ran
            
def print_values(dict_np, idx = 0):
    def printing(l):
        a=''
        for j in l:
            a+=str(j)+'\t'
        print(a)
    printing(dict_np['enc_inp'][idx])
    printing(dict_np['target'][idx])
    printing(dict_np['dec_out'][idx])            

