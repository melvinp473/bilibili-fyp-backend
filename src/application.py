import argparse
import pprint

import dotenv


dotenv.load_dotenv()

from src.app import create_app

application = create_app(debug=True)

if __name__ == '__main__':

    application.run()