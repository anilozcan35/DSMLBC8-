import asyncio
import time
import datetime

from binance import AsyncClient, BinanceSocketManager

from boto3.session import Session

ACCESS_KEY = 'AKIAQZY3J4S4EZD4YKXA'
SECRET_KEY = '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'
bucket_name = 'miuul'

session = Session(aws_access_key_id=ACCESS_KEY,
                  aws_secret_access_key=SECRET_KEY)
s3 = session.resource('s3')

bucket = s3.Bucket(bucket_name)


def upload_file(local_path, remote_path):
    s3.meta.client.upload_file(Filename=local_path, Bucket=bucket_name, Key=remote_path)


async def main():
    active_file_time = int(time.time() / 60)
    f = open("/tmp/" + str(int(active_file_time * 60)) + '.tsv', 'w')
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    # start any sockets here, i.e a trade socket
    ts = bm.trade_socket('BTCUSDT')
    # then start receiving messages
    async with ts as tscm:
        while True:
            res = await tscm.recv()
            new_file_time = int(res['T'] / (1000 * 60))
            if new_file_time != active_file_time:
                upload_file("/tmp/" + str(active_file_time*60) + '.tsv', "data5/" + str(active_file_time*60) + '.tsv')
                active_file_time = new_file_time
                f.close()
                f = open("/tmp/" + str(int(active_file_time * 60)) + '.tsv', 'w')
                print(" #" * 50)
                print(" #" * 50)
                print(" #" * 50)
                print(" #" * 20 + " new file:" + str(int(active_file_time * 60)) + " #" * 20)
                print(" #" * 50)
                print(" #" * 50)
                print(" #" * 50)

            timestamp = f"{datetime.datetime.fromtimestamp(int(res['T'] / 1000)):%Y-%m-%d %H:%M:%S}"
            maker = "0"
            if res['m']:
                maker = "1"

            line = str(res['t']) + "\t" + str(res['s']) + "\t" + '{:.2f}'.format(round(float(res['p']), 2)) + "\t" + str(res['q'])[0:-3] + "\t" + str(
                timestamp) + "\t" + str(maker)
            f.write(line + '\n')
            print(line)
            # print(res)

    await client.close_connection()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())

#
#
# {
#   "e": "trade",     // Event type
#   "E": 123456789,   // Event time
#   "s": "BNBBTC",    // Symbol
#   "t": 12345,       // Trade ID
#   "p": "0.001",     // Price
#   "q": "100",       // Quantity
#   "b": 88,          // Buyer order ID
#   "a": 50,          // Seller order ID
#   "T": 123456785,   // Trade time
#   "m": true,        // Is the buyer the market maker?
#   "M": true         // Ignore
# }
