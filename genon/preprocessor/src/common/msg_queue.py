from typing import Union

import pika
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from pika.exceptions import StreamLostError, ChannelWrongStateError, AMQPChannelError, AMQPConnectionError
import time

from common.logger import Logger
from common.settings import msg_queue_config


class MsgQueue:
    def __init__(self, exchange_name: str, queue_name: str, queue_bind_routing_key: str = '*.*'):
        self._host = msg_queue_config.MQ_HOST
        self._port = msg_queue_config.MQ_PORT
        self._user = msg_queue_config.MQ_USER
        self._password = msg_queue_config.MQ_PASSWORD
        self._vhost = msg_queue_config.MQ_VHOST
        self._exchange_name = exchange_name
        self._exchange_type = msg_queue_config.MQ_EXCHANGE_TYPE
        self._queue_name = queue_name
        self._queue_bind_routing_key = queue_bind_routing_key

        self._connection: Union[BlockingConnection, None] = None
        self._channel: Union[BlockingChannel, None] = None

        self._logger = Logger.getLogger(__name__)

        for i in range(20):
            try:
                self._connect()
                self._logger.info(f'MQ Connected.')
                break

            except Exception as e:
                self._logger.info(f"[{i + 1}/20] Broker Connection Failed. \n\n{repr(e)}\n\n Retrying...")
                time.sleep(5)
                continue
        else:
            self._logger.error('Failed to Connect!')

        return

    def __exit__(self):
        if self._connection and self._connection.is_open:
            self._connection.close()
            self._channel = None

    def _connect(self):

        credentials = pika.PlainCredentials(self._user, self._password)
        parameters = pika.ConnectionParameters(
            host=self._host,
            port=self._port,
            virtual_host=self._vhost,
            credentials=credentials,
            heartbeat=0
        )
        connection = pika.BlockingConnection(parameters=parameters)

        channel = connection.channel()
        channel.exchange_declare(
            exchange=self._exchange_name,
            exchange_type=self._exchange_type
        )
        channel.queue_declare(
            queue=self._queue_name,
            exclusive=False,
            durable=True,
            auto_delete=False,
        )
        channel.queue_bind(
            exchange=self._exchange_name,
            queue=self._queue_name,
            routing_key=self._queue_bind_routing_key
        )

        self._connection = connection
        self._channel = channel

        return channel

    def consume(self, on_message):

        while True:
            try:
                self._channel.basic_consume(
                    queue=self._queue_name,
                    on_message_callback=on_message,
                    auto_ack=True,
                )

                try:
                    self._channel.start_consuming()
                except KeyboardInterrupt:
                    self._channel.stop_consuming()
                    self._connection.close()
                    break

            # Do not recover on channel errors
            except AMQPChannelError as e:
                self._logger.error(f'AMQPChannelError: {repr(e)}. Stop Consuming...')
                break

            # Recover on all other connection errors
            except AMQPConnectionError as e:
                self._logger.error(f'AMQPConnectionError: {repr(e)}')
                self._logger.info(f'Connection was closed. retrying...')
                self._connect()
                continue

            # Do not recover on other errors
            except Exception as e:
                self._logger.error(f'NOT AMQP ERROR: {repr(e)}. Stop Consuming...')
                break

    def publish(self, routing_key, body):

        try:
            self._channel.basic_publish(
                exchange=self._exchange_name,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent,
                )
            )

        except StreamLostError:
            self._connect()
            self._channel.basic_publish(
                exchange=self._exchange_name,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent,
                )
            )

        except ChannelWrongStateError:
            self._connect()
            self._channel.basic_publish(
                exchange=self._exchange_name,
                routing_key=routing_key,
                body=body,
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent,
                )
            )

        return
