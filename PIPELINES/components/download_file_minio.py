from kfp.components import OutputPath

def DownloadFileMinio(
    path_minio,
    access_key,
    secret_key,
    bucket_name,
    filename,
    downloaded_path: OutputPath(str)
):
    
    from minio import Minio
    
    client = Minio(
        path_minio,
        access_key=access_key,
        secret_key=secret_key,
        secure = False
    )

    client.fget_object(bucket_name, filename, downloaded_path)

    

