"""Base class for API clients."""

from dataclasses import dataclass, field
import functools
import os
from posixpath import join
import time
from typing import Callable, Dict, Optional

import requests
from requests.adapters import HTTPAdapter, Retry

# TODO: temporary - for QCIEN-1142 (remove after we stop using verify=False on requests
from requests.packages.urllib3.exceptions import (  # pylint: disable=import-error,no-member
    InsecureRequestWarning,
)

requests.packages.urllib3.disable_warnings(  # pylint: disable=no-member
    InsecureRequestWarning
)

# We are uploading files we want to retry when we receive certain error codes
RETRY_TOTAL = 7
BACKOFF_FACTOR = 2
STATUS_FORCELIST = [502, 503, 504]


@dataclass
class BaseApi:  # pylint: disable=too-many-instance-attributes
    """Base class for API clients."""

    # Hide sensistive info to prevent accidental logging when printing client objects.
    _bearer_info: dict = field(default_factory=dict, repr=False)
    api_token: Optional[str] = field(default=None, repr=False)
    set_bearer_token_on_init: bool = True

    # User-definable url's and endpoints
    url: Optional[str] = None
    authorize: Optional[str] = None

    # Default endpoints and url
    _authorize: str = "authorize"
    _download: str = "contents"
    _add_headers: Dict[str, str] = None

    # Constants
    _user_not_authorized: str = "user not authorized"

    # Request timeout in seconds for connection & read. None for infinite timeout.
    timeout: Optional[float] = 5 * 60.0

    debug: bool = False

    def __post_init__(self):
        self.api_token = self.api_token if self.api_token else os.getenv("QCI_TOKEN")
        assert (
            self.api_token is not None
        ), "QCI_TOKEN environment variable is empty. Specify api_token or add the necessary environment variable"

        self.url = self.url if self.url else os.getenv("QCI_API_URL")
        assert (
            self.url is not None
        ), "QCI_API_URL environment variable is empty. Specify url or add the necessary environment variable"
        # removing trailling / so can add paths simply
        self.url.rstrip("/")

        self.authorize = self.authorize if self.authorize else self._authorize
        self.session = requests.Session()
        retries = Retry(
            total=RETRY_TOTAL,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=STATUS_FORCELIST,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        if self.set_bearer_token_on_init:
            self.set_bearer_token()

    @property
    def auth_url(self) -> str:
        """Return the URL used for authorization."""

        return join(self.url, self.authorize)

    @property
    def headers_without_token(self):
        """Headers without cached bearer token."""
        headers = {
            "Content-Type": "application/json",
            "Connection": "close",
        }

        if self.timeout is not None:
            # Provide client's request timeout to server (latency provides some buffer).
            headers.update({"X-Request-Timeout-Nano": str(int(10**9 * self.timeout))})

        # allow additional headers to be passed through
        if self._add_headers is not None:
            headers.update(self._add_headers)

        return headers

    @property
    def headers(self):
        """Headers with cached bearer token."""
        headers = self.headers_without_token
        headers["Authorization"] = f"Bearer {self._bearer_info.get('access_token', '')}"

        return headers

    @property
    def headers_without_connection_close(self):
        """Headers with cached bearer token, but without connection closing."""
        headers = self.headers
        headers.pop("Connection", None)

        return headers

    def _check_response_error(self, response: requests.Response) -> None:
        """
        Single place to update error check and message for API calls
        :param response: a response from any API call using the requests package
        """
        assert (
            response.status_code < 300
        ), f"Error: {response.text}. Received error code {response.status_code}"

    def get_bearer_token(self) -> requests.Response:
        """Request new bearer token. (Not cached here, see set_bearer_token.)"""
        payload = {"access_token": self.api_token}
        response = self.session.request(
            "POST",
            self.auth_url,
            json=payload,
            headers=self.headers_without_token,
            timeout=self.timeout,
            verify=False,
        )

        self._check_response_error(response)

        return response

    def set_bearer_token(self) -> None:
        """Set bearer token from request."""
        resp = self.get_bearer_token()
        self._bearer_info = resp.json()

    def is_bearer_token_expired(self) -> bool:
        """
        Is current time > 'expires' time.
        TODO: expires_in should be 'expires_at'
        """
        # TODO: this should eventually read 'expires_at'
        # adding 10 second buffer for expiration
        return int(time.time()) + 10 >= self._bearer_info.get("expires_in", 0)

    @staticmethod
    def refresh_token(func) -> Callable:
        """Return a wrapper function that can check an auth token."""

        @functools.wraps(func)
        def check_token(api, *args, **kwargs):
            # Because the decorated function is receiving 'self', we need to pass this
            # additional argument along in the 'api' arg.
            is_expired = api.is_bearer_token_expired()
            # expired, reset the token
            if is_expired:
                api.set_bearer_token()
                return func(api, *args, **kwargs)
            # still have time on the token, so just pass the wrapped func through
            return func(api, *args, **kwargs)

        return check_token
