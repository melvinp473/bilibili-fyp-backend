import dotenv
from src.app import create_app
dotenv.load_dotenv()


application = create_app(debug=True)

if __name__ == '__main__':

    application.run()