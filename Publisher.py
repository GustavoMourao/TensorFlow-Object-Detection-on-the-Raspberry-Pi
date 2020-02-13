from twilio.rest import Client
import os


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
        self.account_sid = os.environ['TWILIO_ACCOUNT_SID']
        self.auth_token = os.environ['TWILIO_AUTH_TOKEN']
        self.my_number = os.environ['MY_DIGITS']
        self.twilio_number = os.environ['TWILIO_DIGITS']

        self.client = Client(
            self.account_sid,
            self.auth_token
        )
