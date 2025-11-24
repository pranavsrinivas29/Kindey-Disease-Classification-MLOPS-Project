import tensorflow as tf
from pathlib import Path


from urllib.parse import urlparse
import shutil
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
import os 


from cnnClassifier.entity.config_entity import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def _valid_generator(self):

        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.30
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )


    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    
    def log_into_mlflow(self):
        import mlflow
        mlflow.set_registry_uri(self.config.mlflow_uri)

        with mlflow.start_run():
            mlflow.set_tag("model_name", "KidneyDiseaseClassifier")
            # Log params and metrics
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )

            # ================================
            # Export model using model.export()
            # ================================
            model_dir = "saved_model"
            
            # Delete previous directory
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

            # Keras 3 → export SavedModel format
            self.model.export(model_dir)

            # ================================
            # Zip SavedModel directory
            # ================================
            zip_path = shutil.make_archive("saved_model", "zip", model_dir)

            # ================================
            # Log artifact to DAGSHub MLflow
            # ================================
            mlflow.log_artifact(zip_path)
