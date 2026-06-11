import json
import pika

# read host from env variable
import os


class RabbitMqProducer:

    def __init__(
        self,
        host=os.getenv("RABBITMQ_HOST", "localhost"),
        queue_name=os.getenv("RABBITMQ_QUEUE", "telemetry")
    ):
        self.host = host
        self.queue_name = queue_name
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=self.host)
        )

        self.channel = self.connection.channel()

        self.channel.queue_declare(
            queue=self.queue_name,
            durable=True
        )

    def publish(self, payload: dict):

        message = json.dumps(payload)

        self.channel.basic_publish(
            exchange="",
            routing_key=self.queue_name,
            body=message,
            properties=pika.BasicProperties(
                delivery_mode=2  # persistent
            )
        )

        print(
          message
        )

    def close(self):

        if self.connection and self.connection.is_open:
            self.connection.close()