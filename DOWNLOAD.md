Dataset **VOST** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/B/c/s4/A8CYlDE9YX4EeH96roVUYR3oAmLpm8ic8bBl0n2hXsqhvsKm7OoA06ZPd56pMQsY7Otxd2gV2jZ7midpqhEjBhNK6OKJo2j0BP0WQ19ztVzOR6g2vwloLezz9pye.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='VOST', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://tri-ml-public.s3.amazonaws.com/datasets/VOST.zip).