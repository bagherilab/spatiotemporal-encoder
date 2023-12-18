import boto3


def download_file(bucket: str, object_path: str, file_name: str) -> None:
    """
    Downloads a file from a given s3 bucket.

    Parameters
    ----------
    bucket : string
        Bucket to download from
    object_path : string
        Full name of file including object path
    file_name: string
        Name to save file as locally

    """
    s3_client = boto3.client("s3")
    try:
        s3_client.download_file(bucket, object_path, file_name)
    except:
        print(f"Could not download {object_path}")
        raise

    print(f"Data from {object_path} successfully downloaded")


def upload_file(file_name: str, bucket: str, object_name: str) -> bool:
    """
    Uploads a local file to an s3 bucket

    Parameters
    ----------
    file_name : string
        File to upload to s3
    bucket : string
        Bucket to upload to
    object_name : string
        Name to store object under in bucket

    Returns
    -------
    boolean
        True on success of upload

    """
    # Establish s3 client
    s3_client = boto3.client("s3")
    response = s3_client.upload_file(file_name, bucket, object_name)

    return response


def list_s3_files(
    bucket: str, object_prefix: str, include: list[str] = [], exclude: list[str] = []
) -> list[str]:
    """
    Lists all files (without downloading) in an s3 object, including all files in nested objects

    Parameters
    ----------
    bucket :
        Bucket to list from
    object_name :
        Name to list from
    include :
        List of substrings on which to filter returned list
    exclude :
        List of substrings on which to filter returned list

    Returns
    -------
    file_list
        List of all files (objects) from specified location

    """
    s3_resource = boto3.resource("s3")
    file_list = []
    my_bucket = s3_resource.Bucket(bucket)

    for object_summary in my_bucket.objects.filter(Prefix=object_prefix):
        file_name = object_summary.key.split("/")[-1]
        file_list.append(file_name)

    if exclude:
        for exclude_term in exclude:
            file_list = list(filter(lambda file: exclude_term not in file, file_list))


    if include:
        final_list = [file for file in file_list if all(term in file for term in include)]
        return list(set(final_list))
    return list(set(file_list))
