from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def fit(self, x, y):
        """_summary_

        Args:
            x (_type_): _description_
            y (_type_): _description_
        """
        pass

    @abstractmethod
    def predict(self, x):
        """_summary_

        Args:
            x (_type_): _description_
        """
        pass
