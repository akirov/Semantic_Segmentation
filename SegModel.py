from abc import ABC, abstractmethod

class SegModel(ABC):

    @abstractmethod
    def train(self, training_data_folder, num_classes, batch_size, num_epochs, save_model_uri):
        pass

    @abstractmethod
    def infer(self, input_images, output_folder, saved_model):
        pass
