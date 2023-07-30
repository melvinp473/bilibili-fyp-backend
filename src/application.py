import argparse
import pprint
import matplotlib.pyplot as plt

import dotenv


dotenv.load_dotenv()

from src.app import create_app

application = create_app(debug=True)

if __name__ == '__main__':

    # Use the "agg" backend to avoid plotting outside the main thread
    plt.switch_backend('agg')

    application.run()
