from anomalib.data import Folder
from anomalib.models import Patchcore
from pytorch_lightning import Trainer


class PatchcoreNoVal(Patchcore):

    def validation_step(self, *args, **kwargs):
        return None

    def validation_epoch_end(self, outputs):
        return None



def main():

    datamodule = Folder(

        root=r"C:\Product_Anomaly_Detection\training\Patch_core\dataset",

        normal_dir="good",

        train_batch_size=1,
        eval_batch_size=1,

        num_workers=0,

        image_size=(256,256),

        task="classification",

    )


    model = PatchcoreNoVal(

        input_size=(256,256),

        backbone="wide_resnet50_2",

        layers=[
            "layer2",
            "layer3"
        ],

        pre_trained=True,

        num_neighbors=9

    )


    trainer = Trainer(

        accelerator="gpu",

        devices=1,

        max_epochs=1,

        num_sanity_val_steps=0,

    )


    trainer.fit(
        model=model,
        datamodule=datamodule
    )


if __name__ == "__main__":
    main()

"""
from anomalib.data import Folder
from anomalib.models import Patchcore
from anomalib.engine import Engine  

def main():

    datamodule = Folder(
        name="ps50_2",
        root=r"C:\Product_Anomaly_Detection\training\Patch_core\dataset",
        normal_dir="train/good",
        num_workers=0,
        train_batch_size=1
    )

    model = Patchcore(
        backbone="wide_resnet50_2", 
        pre_trained=True
    )
    """""" #backbone
    "resnet18"
    "resnet34"
    "resnet50"
    "resnet101"
    "resnet152"
    "resnext50_32x4d"
    "resnext101_32x8d"
    "wide_resnet50_2"
    "wide_resnet101_2"
    """"""
    engine = Engine()

    engine.fit(datamodule=datamodule, model=model)

if __name__ == "__main__":
    main()
"""