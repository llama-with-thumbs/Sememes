"""File storage abstraction — local filesystem (default) or MinIO via env vars."""

import io
import os

STORAGE_BACKEND = os.environ.get("STORAGE_BACKEND", "local")

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
LIBRARY_FOLDER = os.path.join(BASE_DIR, "library")
ATTACHMENTS_FOLDER = os.path.join(BASE_DIR, "attachments")

# Ensure local dirs exist
for d in [UPLOAD_FOLDER, LIBRARY_FOLDER, ATTACHMENTS_FOLDER]:
    os.makedirs(d, exist_ok=True)

_minio_client = None


def _get_minio():
    global _minio_client
    if _minio_client is None:
        from minio import Minio
        _minio_client = Minio(
            os.environ.get("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
        )
    return _minio_client


def _bucket(folder_type):
    """Map folder type to MinIO bucket name."""
    return {
        "library": os.environ.get("MINIO_BUCKET_LIBRARY", "sememes-library"),
        "attachments": os.environ.get("MINIO_BUCKET_ATTACHMENTS", "sememes-attachments"),
        "uploads": os.environ.get("MINIO_BUCKET_UPLOADS", "sememes-uploads"),
    }.get(folder_type, "sememes-uploads")


def _local_folder(folder_type):
    return {"library": LIBRARY_FOLDER, "attachments": ATTACHMENTS_FOLDER, "uploads": UPLOAD_FOLDER}[folder_type]


def save_file(folder_type, filename, file_obj=None, filepath=None):
    """Save a file. Provide either file_obj (file-like) or filepath (path to existing file)."""
    if STORAGE_BACKEND == "minio":
        client = _get_minio()
        bucket = _bucket(folder_type)
        if filepath:
            client.fput_object(bucket, filename, filepath)
        elif file_obj:
            data = file_obj.read()
            client.put_object(bucket, filename, io.BytesIO(data), len(data))
        return f"minio://{bucket}/{filename}"
    else:
        dest = os.path.join(_local_folder(folder_type), filename)
        if filepath:
            import shutil
            shutil.copy2(filepath, dest)
        elif file_obj:
            file_obj.save(dest)
        return dest


def get_file_path(folder_type, filename):
    """Get local file path. For MinIO, downloads to temp first."""
    if STORAGE_BACKEND == "minio":
        import tempfile
        client = _get_minio()
        bucket = _bucket(folder_type)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        client.fget_object(bucket, filename, tmp.name)
        return tmp.name
    else:
        return os.path.join(_local_folder(folder_type), filename)


def get_file_bytes(folder_type, filename):
    """Get file content as bytes."""
    if STORAGE_BACKEND == "minio":
        client = _get_minio()
        bucket = _bucket(folder_type)
        response = client.get_object(bucket, filename)
        data = response.read()
        response.close()
        response.release_conn()
        return data
    else:
        path = os.path.join(_local_folder(folder_type), filename)
        with open(path, "rb") as f:
            return f.read()


def delete_file(folder_type, filename):
    """Delete a file."""
    if STORAGE_BACKEND == "minio":
        client = _get_minio()
        bucket = _bucket(folder_type)
        try:
            client.remove_object(bucket, filename)
        except Exception:
            pass
    else:
        path = os.path.join(_local_folder(folder_type), filename)
        if os.path.exists(path):
            os.remove(path)


def list_files(folder_type, prefix=""):
    """List files in a folder."""
    if STORAGE_BACKEND == "minio":
        client = _get_minio()
        bucket = _bucket(folder_type)
        return [obj.object_name for obj in client.list_objects(bucket, prefix=prefix)]
    else:
        folder = _local_folder(folder_type)
        return [f for f in os.listdir(folder) if f.startswith(prefix)] if prefix else os.listdir(folder)


def file_exists(folder_type, filename):
    """Check if a file exists."""
    if STORAGE_BACKEND == "minio":
        client = _get_minio()
        bucket = _bucket(folder_type)
        try:
            client.stat_object(bucket, filename)
            return True
        except Exception:
            return False
    else:
        return os.path.exists(os.path.join(_local_folder(folder_type), filename))


def get_file_size(folder_type, filename):
    """Get file size in bytes."""
    if STORAGE_BACKEND == "minio":
        client = _get_minio()
        bucket = _bucket(folder_type)
        stat = client.stat_object(bucket, filename)
        return stat.size
    else:
        return os.path.getsize(os.path.join(_local_folder(folder_type), filename))


def is_minio():
    """Check if using MinIO storage."""
    return STORAGE_BACKEND == "minio"
