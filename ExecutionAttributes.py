import datetime
from keras.preprocessing.image import ImageDataGenerator


class ExecutionAttribute:

    def __init__(self, summ_basename="", img_width=0, img_height=0, path="", epochs=0, batch_size=0, train_generator=None, validation_generator=None, test_generator=None, model=None, seq=0, architecture="", csv_path="", fusion="None", fnames_test=None, labels_test=None, npy_path=None):
        self.img_width = img_width
        self.img_height = img_height
        self.path = path
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.test_generator = test_generator
        self.summ_basename = summ_basename
        self.train_data_dir = path + '/train'
        self.validation_data_dir = path + '/valid'
        self.test_data_dir = path + '/test'
        self.steps_train = 0
        self.steps_valid = 0
        self.steps_test = 0
        self.seq = seq
        self.architecture = architecture
        self.curr_basename = self.summ_basename + '-' + str(self.seq)
        self.init_timestamp = datetime.datetime.now()
        self.csv_path = csv_path
        self.fusion = fusion
        self.train_samples = 0
        self.valid_samples = 0
        self.test_samples = 0
        self.fnames_test = fnames_test
        self.labels_test = labels_test
        self.numpy_path = npy_path

    def calculate_steps(self):
        if self.fusion != "None":
            self.steps_train = self.train_samples // self.batch_size
            self.steps_valid = self.valid_samples // self.batch_size
            self.steps_test = self.test_samples // 1 
        else:
            self.steps_train = self.train_generator.n // self.train_generator.batch_size
            self.steps_valid = self.validation_generator.n // self.validation_generator.batch_size
            self.steps_test = self.test_generator.samples // self.test_generator.batch_size

    def set_dir_names(self):
        self.train_data_dir = self.path + '/train'
        self.validation_data_dir = self.path + '/valid'
        self.test_data_dir = self.path + '/test'

    def increment_seq(self):
        self.seq = self.seq + 1
        self.curr_basename = self.summ_basename + '-' + str(self.seq)

    def calculate_samples_len(self):
        gen = ImageDataGenerator()
        transient_train_generator = gen.flow_from_directory(self.train_data_dir, shuffle=False,
                                                                      batch_size=self.batch_size, seed=666)
        transient_validation_generator = gen.flow_from_directory(self.validation_data_dir, shuffle=False,
                                                                          batch_size=self.batch_size, seed=666)
        transient_test_generator = gen.flow_from_directory(self.test_data_dir, shuffle=False,
                                                                    batch_size=self.batch_size, seed=666)

        self.train_samples = len(transient_train_generator.filenames)
        self.valid_samples = len(transient_validation_generator.filenames)
        self.test_samples = len(transient_test_generator.filenames)

        self.labels_test = transient_test_generator.classes
        self.fnames_test = transient_test_generator.filenames

