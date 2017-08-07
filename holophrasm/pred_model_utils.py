import numpy as np

max_main_statement_length = 0
max_main_hyps_length = 0
max_prop_statement_length = 0
max_prop_hyps_length = 0


class Config:
    def __init__(self, language_model):
        self.input_dim = 128
        self.output_dim = 128
        self.special_symbols = ['END_OF_HYP', 'END_OF_SECTION', 'START_OUTPUT', 'TARGET_UC', 'UC']
        self.negative_samples = 4
        self.lm = language_model
        self.construct_dictionary(self.lm)
        self.matrix_init = self.make_matrix([self.num_tokens, self.input_dim])
        self.matrix_weight =self.make_matrix([self.output_dim, self.output_dim])

    def construct_dictionary(self, language_model):
        constructor_label_to_number = language_model.constructor_label_to_number
        decode = [None] * len(constructor_label_to_number)
        for label in constructor_label_to_number:
            number = constructor_label_to_number[label]
            assert decode[number] is None
            decode[number] = label
        decode = decode + self.special_symbols
        self.dic={}
        for i in range(len(decode)):
            self.dic[decode[i]] = i
        self.num_tokens = len(self.dic)
        print(('Config(): added '+str(self.num_tokens)+' tokens to dictionary'))
    
    def make_matrix(self, shape):
        total = np.sum(shape)
        assert total != 0
        x = (6.0/total)**0.5
        return np.random.uniform(low=-x, high=x, size=shape).astype(np.float64)
    
    def encode(self, token, structure_data=None):
        index = self.dic[token]
        out = np.concatenate((self.matrix_init[index],structure_data))
        return out

    def encode_string(self, string, structure_datas = None):
        return [self.encode(token, structure_data = sd) for token, sd in zip(string, structure_datas)]
    
    def get_props(self, proof_step):
        wrong_props = self.negative_samples
        labels = self.lm.searcher.search(proof_step.tree, proof_step.context, max_proposition=proof_step.context.number, vclass='|-')
        labels.remove(proof_step.prop.label)
        #wrong_props = min(wrong_props, len(labels))
        #rand = np.random.choice(len(labels), wrong_props, replace=False)
        rand = np.random.choice(len(labels), wrong_props, replace=True)
        rc = [labels[i] for i in rand]
        wrong_props = [self.lm.database.propositions[label] for label in rc]
        return [proof_step.prop]+wrong_props
    
    def parse_statement_and_hyps(self, statement, hyps, f, typ = 'main'):
        random_replacement_dict = self.lm.random_replacement_dict_f(f=f)
        statement = statement.copy().replace_values(random_replacement_dict)
        hyps = [h.tree.copy().replace_values(random_replacement_dict) for h in hyps if h.type=='e']
        statement_graph_structure = TreeInformation([statement],
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        hyps_graph_structure = TreeInformation(hyps,
                start_symbol=None, intermediate_symbol='END_OF_HYP',
                end_symbol='END_OF_SECTION')
        if (typ == 'main'):
            global max_main_statement_length
            global max_main_hyps_length
            max_main_statement_length = max(max_main_statement_length, len(statement_graph_structure.string))
            max_main_hyps_length = max(max_main_hyps_length, len(hyps_graph_structure.string))
        elif (typ == 'prop'):
            global max_prop_statement_length
            global max_prop_hyps_length
            max_prop_statement_length = max(max_prop_statement_length, len(statement_graph_structure.string))
            max_prop_hyps_length = max(max_prop_hyps_length, len(hyps_graph_structure.string))
        return merge_graph_structures([statement_graph_structure, hyps_graph_structure])

def get_max_main_statement_length():
    global max_main_statement_length
    return max_main_statement_length

def get_max_main_hyps_length():
    global max_main_hyps_length
    return max_main_hyps_length

def get_max_prop_statement_length():
    global max_prop_statement_length
    return max_prop_statement_length

def get_max_prop_hyps_length():
    global max_prop_hyps_length
    return max_prop_hyps_length

def pad_sequence(training_data, length, typ = 'main'):
    
    max_statement_length = 0
    max_hyps_length = 0
    
    if typ == 'main':
        global max_main_statement_length
        global max_main_hyps_length
        max_statement_length = max_main_statement_length
        max_hyps_length = max_main_hyps_length
        
    elif typ == 'prop':
        global max_prop_statement_length
        global max_prop_hyps_length
        max_statement_length = max_prop_statement_length
        max_hyps_length = max_prop_hyps_length
        
    shape = np.shape(training_data)
    
    training_data_statement = training_data[0:length]
    padding_statement = np.zeros([max_statement_length-length,shape[1]])
        
    training_data_hyps = training_data[length:]
    padding_hyps= np.zeros([max_hyps_length-shape[0]+length,shape[1]])
   
    training_data = np.concatenate((training_data_statement, padding_statement, training_data_hyps, padding_hyps))
    
    return training_data

def merge_graph_structures(gs_list):
    out_string = []
    out_parents = []
    out_left_siblings = []
    out_right_siblings = []
    out_depth = []
    out_parent_arity = []
    out_leaf_position = []
    out_arity = []
    length = 0

    for gs in gs_list:
        current_n = len(out_string)
        out_string += gs.string
        out_parents += [(-1 if x==-1 else x+current_n) for x in gs.parents]
        out_left_siblings += [(-1 if x==-1 else x+current_n) for x in gs.left_sibling]
        out_right_siblings += [(-1 if x==-1 else x+current_n) for x in gs.right_sibling]
        out_depth += gs.depth
        out_parent_arity += gs.parent_arity
        out_leaf_position += gs.leaf_position
        out_arity += gs.arity
        length += (len(gs.string) if length == 0 else 0)

    return out_string, out_parents, out_left_siblings, out_right_siblings, \
            out_depth, out_parent_arity, out_leaf_position, out_arity, length


class TreeInformation:
    def __init__(self, trees, start_symbol=None,
            intermediate_symbol=None, end_symbol=None):
        self.parents = []
        self.left_sibling = []
        self.right_sibling = []
        self.string = []

        self.depth = []
        self.parent_arity = []
        self.leaf_position = []
        self.arity = []

        self.n=0

        if start_symbol is not None:
            self.right_sibling.append(-1)
            self.parents.append(-1)
            self.left_sibling.append(-1)
            self.string.append(start_symbol)

            self.depth.append(-1)
            self.parent_arity.append(-1)
            self.leaf_position.append(-1)
            self.arity.append(-1)

            self.n+=1

        for i, tree in enumerate(trees):
            self.add_tree(tree)
            self.add_tree_right_siblings(tree)

            if i is not len(trees)-1 and intermediate_symbol is not None:
                self.right_sibling.append(-1)
                self.parents.append(-1)
                self.left_sibling.append(-1)
                self.string.append(intermediate_symbol)

                self.depth.append(-1)
                self.parent_arity.append(-1)
                self.leaf_position.append(-1)
                self.arity.append(-1)

                self.n+=1

        if end_symbol is not None:
            self.right_sibling.append(-1)
            self.parents.append(-1)
            self.left_sibling.append(-1)
            self.string.append(end_symbol)

            self.depth.append(-1)
            self.parent_arity.append(-1)
            self.leaf_position.append(-1)
            self.arity.append(-1)

            self.n+=1

        # verify some stuff
        length = len(self.string)
        assert len(self.right_sibling) == length
        assert len(self.parents) == length
        assert len(self.left_sibling) == length
        assert len(self.depth) == length
        assert len(self.parent_arity) == length
        assert len(self.leaf_position) == length
        assert len(self.arity) == length

    def params(self):
        return self.string, self.parents, self.left_sibling, self.right_sibling, \
                self.depth, self.parent_arity, self.leaf_position, self.arity

    def add_tree(self, tree, parent=-1, left_sibling=-1, depth=0, parent_arity=-1, leaf_position=-1):
        degree = len(tree.leaves)
        this_n = self.n
        tree.ti_index = this_n
        self.parents.append(parent)
        self.left_sibling.append(left_sibling)
        self.string.append(tree.value)
        self.depth.append(depth)
        self.parent_arity.append(parent_arity)
        self.leaf_position.append(leaf_position)
        arity = len(tree.leaves)
        self.arity.append(arity)
        self.n += 1

        prev_n = -1
        for i, c in enumerate(tree.leaves):
            self.add_tree(c, parent=this_n, left_sibling=prev_n, depth=depth+1, parent_arity=arity, leaf_position=i)
            prev_n=c.ti_index

    def add_tree_right_siblings(self, tree, right_sibling = -1):
        self.right_sibling.append(right_sibling)
        degree = len(tree.leaves)

        for i,c in enumerate(tree.leaves):
            if i < degree-1:
                next_right = tree.leaves[i+1].ti_index
            else:
                next_right = -1
            self.add_tree_right_siblings(c, next_right)
