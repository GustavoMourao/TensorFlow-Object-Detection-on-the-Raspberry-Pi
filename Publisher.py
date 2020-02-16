from twilio.rest import Client
import os
from dotenv import Dotenv


class Publisher:
    """
    Publisher service.
    """
    def __init__(self):
        """
        Initialize Publisher object.
        Twilio SID, authentication token, my phone number
        and the Twilio phone number
        are stored as environment variables on my Pi
        """
        dotenv = Dotenv('.env')
        self.account_sid = dotenv['TWILIO_ACCOUNT_SID']
        self.auth_token = dotenv['TWILIO_AUTH_TOKEN']
        self.my_number = dotenv['MY_DIGITS']
        self.twilio_number = dotenv['TWILIO_DIGITS']

        self.client = Client(
            self.account_sid,
            self.auth_token
        )

    def send_person_detection(self):
        """
        Send person detection status
        """
        self.client.messages.create(
            body='New person detected!',
            from_=self.twilio_number,
            to=self.my_number
        )
