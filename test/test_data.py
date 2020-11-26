import pickle

AUDIO = b'covarep'
VISUAL = b'facet'
TEXT = b'glove'
LABEL = b'label'
TRAIN = b'train'
VALID = b'valid'
TEST = b'test'

pom_data = pickle.load(open("../data/pom.pkl", 'rb'),encoding='bytes')

pom_train, pom_valid, pom_test = pom_data[TRAIN], pom_data[VALID], pom_data[TEST]

train_audio, train_visual, train_text, train_labels \
        = pom_train[AUDIO], pom_train[VISUAL], pom_train[TEXT], pom_train[LABEL]
valid_audio, valid_visual, valid_text, valid_labels \
        = pom_valid[AUDIO], pom_valid[VISUAL], pom_valid[TEXT], pom_valid[LABEL]
test_audio, test_visual, test_text, test_labels \
        = pom_test[AUDIO], pom_test[VISUAL], pom_test[TEXT], pom_test[LABEL]
a = 1
