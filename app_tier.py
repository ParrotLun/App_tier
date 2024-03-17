# app_tier.py
import logging
import torch
from PIL import Image
from io import BytesIO
from facenet_pytorch import MTCNN, InceptionResnetV1
import base64
import boto3
import json
import time

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化MTCNN和InceptionResnet
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# 初始化 SQS 客戶端
sqs= boto3.client('sqs', region_name='us-east-1')
req_queue_url = 'https://sqs.us-east-1.amazonaws.com/905418311239/1229850065-req-queue'
resp_queue_url = 'https://sqs.us-east-1.amazonaws.com/905418311239/1229850065-resp-queue'

s3 = boto3.client('s3', region_name='us-east-1')
input_bucket = '1229850065-in-bucket'
output_bucket = '1229850065-out-bucket'

def face_match(image, data_path='data.pt'):
    face, prob = mtcnn(image, return_prob=True)
    if face is not None and prob > 0.9:
        emb = resnet(face.unsqueeze(0)).detach()
        saved_data = torch.load(data_path)
        embedding_list, name_list = saved_data[0], saved_data[1]
        dist_list = [torch.dist(emb, emb_db).item() for emb_db in embedding_list]
        idx_min = dist_list.index(min(dist_list))
        return name_list[idx_min]
    return "Unknown", None


def upload_image_to_s3(image_data, filename):
    """
    上传图片到S3输入桶。
    """
    try:
        s3.upload_fileobj(BytesIO(image_data), input_bucket, filename)
        logging.info(f"Image {filename} uploaded to {input_bucket}")
    except Exception as e:
        logging.error(f"Failed to upload image to S3: {e}")


def upload_result_to_s3(result, filename):
    """
    上传识别结果到S3输出桶。
    """
    try:
        result_bytes = result.encode('utf-8')
        s3.put_object(Bucket=output_bucket, Key=filename, Body=result_bytes)
        logging.info(f"Result for {filename} uploaded to {output_bucket}")
    except Exception as e:
        logging.error(f"Failed to upload result to S3: {e}")


def check_queue_messages(queue_url):
    # 檢查隊列中的消息數量
    attrs = sqs.get_queue_attributes(
        QueueUrl=queue_url,
        AttributeNames=['ApproximateNumberOfMessages']
    )
    num_messages = int(attrs['Attributes']['ApproximateNumberOfMessages'])
    return num_messages > 0

def process_messages():
    while True:
        if check_queue_messages(req_queue_url):
            resp = sqs.receive_message(
                QueueUrl=req_queue_url,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
            )
            messages = resp.get('Messages', [])
            if not messages:
                continue

            for message in messages:
                try:
                    body = json.loads(message['Body'])
                    img_data = base64.b64decode(body['image_data'])
                    
                    filename = body['filename']
                    upload_image_to_s3(img_data, filename)

                    img = Image.open(BytesIO(img_data))
                    name = face_match(img)

                    upload_result_to_s3(name, filename.split('.')[0])  # 除去文件扩展名

                    response_data = {
                        'request_id': body['request_id'],
                        'filename': body['filename'],
                        'classification_result': name,
                    }
                    print(response_data)
                    sqs.send_message(
                        QueueUrl=resp_queue_url,
                        MessageBody=json.dumps(response_data)
                    )
                except Exception as e:
                    logging.error(f"Error processing message: {e}")
                finally:
                    print("Message deleted from requests queue")
                    sqs.delete_message(
                        QueueUrl=req_queue_url,
                        ReceiptHandle=message['ReceiptHandle']
                    )
        else:
            print("No messages in queue, waiting...")
            time.sleep(1)  # 等待一段時間後再次檢查隊列

if __name__ == '__main__':
    logging.info("Starting message processing script...")
    process_messages()
