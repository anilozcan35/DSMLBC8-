import boto3
from boto3.session import Session

ACCESS_KEY = 'AKIATMDCCZVV5IYNMPWI'
SECRET_KEY = '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'
bucket_name = 'bucketofmiuul'

session = Session(aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)
s3 = session.resource('s3')

bucket = s3.Bucket(bucket_name)


def list_bucket():
    for s3_file in bucket.objects.all():
        print(s3_file.key)

def download_file():
    bucket.download_file('year_2010_2011.tsv', 'AWS/yeni_data.tsv')

def upload_file():
    local_path = 'AWS/ec2-ug.pdf'
    bucket_path = 'pdf/denemecik.pdf'

    s3.meta.client.upload_file(Filename=local_path, Bucket=bucket_name, Key=bucket_path)

def delete_file():
    s3.Object(bucket_name, 'mydata/titanic.csv').delete()


def copy_file():
    copy_source = {
        'Bucket': 'itisrainingman',
        'Key': 'year_2009_2010_clean.tsv'
    }

    bucket.copy(copy_source, 'copy/year_2009_2010_clean.tsv')

    print(' File is copied')


copy_file()