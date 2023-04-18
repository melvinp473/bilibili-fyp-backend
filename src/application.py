import argparse
import dotenv
dotenv.load_dotenv()

from src.app import create_app

application = create_app(debug=True)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(
    #     description=__doc__,
    #     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument('--host')
    # parser.add_argument('--port')
    #
    # args = parser.parse_args()
    # print('args', args)
    # host = None
    # port = None
    # if args.host:
    #     host = args.host
    # if args.port:
    #     port = args.port

    application.run()