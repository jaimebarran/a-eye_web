import logging

from authlib.integrations.flask_client import OAuth
from flask import Flask
from flask_mail import Mail
from werkzeug.middleware.proxy_fix import ProxyFix

from config import LOGS_FOLDER, Config

mail = Mail()
oauth = OAuth()


def create_app():
    app = Flask(__name__, static_folder="static")
    app.config.from_object(Config)

    app.secret_key = app.config["SECRET_KEY"]
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    # Setup logging
    logging.basicConfig(
        filename=f"{LOGS_FOLDER}/app.log",
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    mail.init_app(app)
    oauth.init_app(app)

    oauth.register(
        "auth0",
        client_id=app.config["AUTH0_CLIENT_ID"],
        client_secret=app.config["AUTH0_CLIENT_SECRET"],
        api_base_url=f"https://{app.config['AUTH0_DOMAIN']}",
        access_token_url=f"https://{app.config['AUTH0_DOMAIN']}/oauth/token",
        authorize_url=f"https://{app.config['AUTH0_DOMAIN']}/authorize",
        client_kwargs={"scope": "openid profile email"},
        server_metadata_url=f"https://{app.config['AUTH0_DOMAIN']}/.well-known/openid-configuration",
    )

    from routes import bp as routes_bp

    app.register_blueprint(routes_bp)

    return app
