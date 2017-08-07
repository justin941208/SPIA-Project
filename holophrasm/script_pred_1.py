import tflearn
import tensorflow as tf
from data_utils5 import LanguageModel
from pred_model_utils import *
from tree_parser import file_contents, meta_math_database

main_statement_time_steps = 100
main_hyps_time_steps = 100
prop_statement_time_steps = 100
prop_hyps_time_steps = 100
input_dim = 132
output_dim = 128
num_props = 5
p = 0.9999 # probability of Dropout keeping node


def prop_net(prop_statement_input, prop_hyps_input):
    with tf.variable_scope('statement'):
        prop_statement_net = tflearn.gru(prop_statement_input, output_dim, return_seq=False, dynamic=True)
        print(prop_statement_net)
    with tf.variable_scope('hyps'):
        prop_net = tflearn.gru(prop_hyps_input, output_dim, return_seq=False, initial_state=prop_statement_net,
                               dynamic=True)
        print(prop_net)
    return prop_net


text = file_contents()
database = meta_math_database(text, n=2000)
language_model = LanguageModel(database)
config = Config(language_model)

training_proof_steps = language_model.training_proof_steps

training_data_main = [None] * len(training_proof_steps)
training_data_main_length = [None] * len(training_proof_steps)
training_data_props = [None] * len(training_proof_steps)
training_data_props_length = [None] * len(training_proof_steps)

for i, proof_step in enumerate(training_proof_steps):

    in_string, in_parents, in_left, in_right, depths, \
    parent_arity, leaf_position, arity, length = config.parse_statement_and_hyps(proof_step.tree,
                                                                                 proof_step.context.hyps,
                                                                                 proof_step.context.f, 'main')
    training_data_main[i] = config.encode_string(in_string, list(zip(depths, parent_arity, leaf_position, arity)))
    training_data_main_length[i] = length

    props = config.get_props(proof_step)
    temp = [None] * len(props)
    temp_length = [None] * len(props)
    for j, prop in enumerate(props):
        in_string, in_parents, in_left, in_right, depths, \
        parent_arity, leaf_position, arity, length = config.parse_statement_and_hyps(prop.tree, prop.hyps, prop.f,
                                                                                     'prop')
        temp[j] = config.encode_string(in_string, list(zip(depths, parent_arity, leaf_position, arity)))
        temp_length[j] = length
    training_data_props[i] = temp
    training_data_props_length[i] = temp_length

for i in range(len(training_data_main)):
    training_data_main[i] = pad_sequence(training_data_main[i], training_data_main_length[i], 'main')

for i in range(len(training_data_props)):
    for j in range(num_props):
        training_data_props[i][j] = pad_sequence(training_data_props[i][j], training_data_props_length[i][j], 'prop')

training_data_main = np.transpose(training_data_main, (1, 0, 2))
training_data_props = np.transpose(training_data_props, (1, 2, 0, 3))

print('Data preprocessing finished')

tf.reset_default_graph()

main_statement_time_steps = get_max_main_statement_length()
main_hyps_time_steps = get_max_main_hyps_length()

prop_statement_time_steps = get_max_prop_statement_length()
prop_hyps_time_steps = get_max_prop_hyps_length()

main_statement_input = tflearn.input_data([None, main_statement_time_steps, input_dim], name='main_statement_input')
main_hyps_input = tflearn.input_data([None, main_hyps_time_steps, input_dim], name='main_hyps_input')

with tf.variable_scope('main') as scope:
    main_statement_net = tflearn.gru(main_statement_input, output_dim, return_seq=False, dynamic=True,
                                     scope='main_statement')
    print(main_statement_net)
    main_net = tflearn.gru(main_hyps_input, output_dim, return_seq=False, initial_state=main_statement_net,
                           dynamic=True, scope='main_hyps')
    print(main_net)
main_net = tflearn.dropout(main_net, p)

prop_statement_input_1 = tflearn.input_data([None, prop_statement_time_steps, input_dim], name='prop_statement_input_1')
prop_hyps_input_1 = tflearn.input_data([None, prop_hyps_time_steps, input_dim], name='prop_hyps_input_1')

prop_statement_input_2 = tflearn.input_data([None, prop_statement_time_steps, input_dim], name='prop_statement_input_2')
prop_hyps_input_2 = tflearn.input_data([None, prop_hyps_time_steps, input_dim], name='prop_hyps_input_2')

prop_statement_input_3 = tflearn.input_data([None, prop_statement_time_steps, input_dim], name='prop_statement_input_3')
prop_hyps_input_3 = tflearn.input_data([None, prop_hyps_time_steps, input_dim], name='prop_hyps_input_3')

prop_statement_input_4 = tflearn.input_data([None, prop_statement_time_steps, input_dim], name='prop_statement_input_4')
prop_hyps_input_4 = tflearn.input_data([None, prop_hyps_time_steps, input_dim], name='prop_hyps_input_4')

prop_statement_input_5 = tflearn.input_data([None, prop_statement_time_steps, input_dim], name='prop_statement_input_5')
prop_hyps_input_5 = tflearn.input_data([None, prop_hyps_time_steps, input_dim], name='prop_hyps_input_5')

with tf.variable_scope('prop') as scope:
    prop_net_1 = prop_net(prop_statement_input_1, prop_hyps_input_1)
    scope.reuse_variables()
    prop_net_2 = prop_net(prop_statement_input_2, prop_hyps_input_2)
    prop_net_3 = prop_net(prop_statement_input_3, prop_hyps_input_3)
    prop_net_4 = prop_net(prop_statement_input_4, prop_hyps_input_4)
    prop_net_5 = prop_net(prop_statement_input_5, prop_hyps_input_5)

main_net = tflearn.fully_connected(main_net, output_dim, weights_init='xavier')
main_net = tflearn.relu(main_net)
main_net = tflearn.fully_connected(main_net, output_dim, weights_init='xavier')
main_net = tflearn.relu(main_net)
main_net = tflearn.dropout(main_net, p)

prop_net_1 = tflearn.fully_connected(prop_net_1, output_dim, weights_init='xavier')
prop_net_1 = tflearn.relu(prop_net_1)
prop_net_1 = tflearn.dropout(prop_net_1, p)
prop_net_1 = tflearn.fully_connected(prop_net_1, output_dim, weights_init='xavier')
prop_net_1 = tflearn.relu(prop_net_1)
prop_net_1 = tflearn.dropout(prop_net_1, p)

prop_net_2 = tflearn.fully_connected(prop_net_2, output_dim, weights_init='xavier')
prop_net_2 = tflearn.relu(prop_net_2)
prop_net_2 = tflearn.dropout(prop_net_2, p)
prop_net_2 = tflearn.fully_connected(prop_net_2, output_dim, weights_init='xavier')
prop_net_2 = tflearn.relu(prop_net_2)
prop_net_2 = tflearn.dropout(prop_net_2, p)

prop_net_3 = tflearn.fully_connected(prop_net_3, output_dim, weights_init='xavier')
prop_net_3 = tflearn.relu(prop_net_3)
prop_net_3 = tflearn.dropout(prop_net_3, p)
prop_net_3 = tflearn.fully_connected(prop_net_3, output_dim, weights_init='xavier')
prop_net_3 = tflearn.relu(prop_net_3)
prop_net_3 = tflearn.dropout(prop_net_3, p)

prop_net_4 = tflearn.fully_connected(prop_net_4, output_dim, weights_init='xavier')
prop_net_4 = tflearn.relu(prop_net_4)
prop_net_4 = tflearn.dropout(prop_net_4, p)
prop_net_4 = tflearn.fully_connected(prop_net_4, output_dim, weights_init='xavier')
prop_net_4 = tflearn.relu(prop_net_4)
prop_net_4 = tflearn.dropout(prop_net_4, p)

prop_net_5 = tflearn.fully_connected(prop_net_5, output_dim, weights_init='xavier')
prop_net_5 = tflearn.relu(prop_net_5)
prop_net_5 = tflearn.dropout(prop_net_5, p)
prop_net_5 = tflearn.fully_connected(prop_net_5, output_dim, weights_init='xavier')
prop_net_5 = tflearn.relu(prop_net_5)
prop_net_5 = tflearn.dropout(prop_net_5, p)

net = tflearn.fully_connected(main_net, output_dim, bias=False, weights_init='xavier')

net_1 = tf.sigmoid(tf.reduce_sum(tf.multiply(net, prop_net_1)))
net_2 = tf.sigmoid(-tf.reduce_sum(tf.multiply(net, prop_net_2)))
net_3 = tf.sigmoid(-tf.reduce_sum(tf.multiply(net, prop_net_3)))
net_4 = tf.sigmoid(-tf.reduce_sum(tf.multiply(net, prop_net_4)))
net_5 = tf.sigmoid(-tf.reduce_sum(tf.multiply(net, prop_net_5)))

net = -tf.log(net_1) - tf.log(net_2) - tf.log(net_3) - tf.log(net_4) - tf.log(net_5)


def my_objective(y_pred, y_true):
    return y_pred


net = tflearn.regression(net, placeholder=None, loss=my_objective, learning_rate=1.0e-5)

model = tflearn.DNN(net, tensorboard_verbose=3)

model.fit(X_inputs={'main_statement_input': np.transpose(training_data_main[:main_statement_time_steps], (1, 0, 2)),
                    'main_hyps_input': np.transpose(training_data_main[main_statement_time_steps:], (1, 0, 2)),
                    'prop_statement_input_1': np.transpose(training_data_props[0][:prop_statement_time_steps],
                                                           (1, 0, 2)),
                    'prop_hyps_input_1': np.transpose(training_data_props[0][prop_statement_time_steps:], (1, 0, 2)),
                    'prop_statement_input_2': np.transpose(training_data_props[1][:prop_statement_time_steps],
                                                           (1, 0, 2)),
                    'prop_hyps_input_2': np.transpose(training_data_props[1][prop_statement_time_steps:], (1, 0, 2)),
                    'prop_statement_input_3': np.transpose(training_data_props[2][:prop_statement_time_steps],
                                                           (1, 0, 2)),
                    'prop_hyps_input_3': np.transpose(training_data_props[2][prop_statement_time_steps:], (1, 0, 2)),
                    'prop_statement_input_4': np.transpose(training_data_props[3][:prop_statement_time_steps],
                                                           (1, 0, 2)),
                    'prop_hyps_input_4': np.transpose(training_data_props[3][prop_statement_time_steps:], (1, 0, 2)),
                    'prop_statement_input_5': np.transpose(training_data_props[4][:prop_statement_time_steps],
                                                           (1, 0, 2)),
                    'prop_hyps_input_5': np.transpose(training_data_props[4][prop_statement_time_steps:], (1, 0, 2))},
          Y_targets=None, n_epoch=30, batch_size=100)

model.save('pred_model.tflearn')

np.save('Pred_W', model.get_weights(tflearn.get_layer_variables_by_name('Weight')[0]))
