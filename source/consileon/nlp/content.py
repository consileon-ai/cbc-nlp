import logging
from abc import ABC, abstractmethod
from pathlib import Path
from os import listdir
from os.path import isfile
import boto3


logger = logging.getLogger('consileon.nlp.content')


class ContentHandler(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def list(self, prefix):
        pass

    @abstractmethod
    def save_text(self, key, text, prefix=""):
        pass

    @abstractmethod
    def get_text(self, key, prefix=""):
        pass

    @classmethod
    def append_prefix(cls, base_prefix, prefix):
        sep = ""
        if len(base_prefix) > 0 and len(prefix) > 0 and not base_prefix.endswith("/") and not prefix.startswith("/"):
            sep = "/"
        return base_prefix + sep + prefix


class FileSystemContentHandler(ContentHandler, ABC):

    def __init__(
        self,
        base_folder='.',
        encoding='utf-8'
    ):
        self.base_folder = base_folder
        self.encoding = encoding
        super().__init__()

    def list(self, prefix=""):
        path = Path(self.base_folder + "/" + prefix)
        path.mkdir(parents=True, exist_ok=True)
        return [f for f in listdir(path) if isfile(path / f)]

    def save_text(self, key, text, prefix=""):
        path = Path(self.base_folder + "/" + prefix)
        path.mkdir(parents=True, exist_ok=True)
        full_path = path / key
        with full_path.open("w", encoding=self.encoding) as f:
            f.write(text)

    def get_text(self, key, prefix=""):
        full_path = self.base_folder + "/" + prefix + "/" + key
        text = Path(full_path).read_text(encoding=self.encoding)
        return text


class AwsS3ContentHandler(ContentHandler, ABC):

    def __init__(
        self,
        bucket=None,
        base_prefix='',
        encoding='utf-8'
    ):
        assert(bucket is not None)
        self.bucket = bucket
        self.base_prefix = base_prefix
        self.encoding = encoding
        self.client = boto3.client('s3')
        super().__init__()

    def list(self, prefix=""):
        result = []
        full_prefix = self.append_prefix(self.base_prefix, prefix)
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)
        replace = ""
        if len(full_prefix) > 0 and not full_prefix.endswith("/"):
            replace = full_prefix + "/"
        while True and response.get("Contents"):
            contents = response["Contents"]
            result.extend([i["Key"].replace(replace, "") for i in contents])
            if not response["IsTruncated"]:
                break
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                ContinuationToken=response['NextContinuationToken'],
                Prefix=full_prefix
            )
        return result

    def save_text(self, key, text, prefix=""):
        full_prefix = self.append_prefix(self.base_prefix, prefix)
        full_key = self.append_prefix(full_prefix, key)
        self.client.put_object(
            Bucket=self.bucket,
            Key=full_key,
            ContentEncoding=self.encoding,
            Body=text.encode(self.encoding)
        )

    def get_text(self, key, prefix=""):
        full_prefix = self.append_prefix(self.base_prefix, prefix)
        full_key = self.append_prefix(full_prefix, key)
        response = self.client.get_object(Bucket=self.bucket, Key=full_key)
        text = response['Body'].read().decode(response['ContentEncoding'])
        return text
