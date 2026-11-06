"""Service-level errors. The API maps them to HTTP statuses; cross-organisation access is always NotFound (404), never Forbidden."""


class ServiceError(Exception):
    status = 400
    code = "error"


class NotFound(ServiceError):
    status = 404
    code = "not_found"


class Forbidden(ServiceError):
    status = 403
    code = "forbidden"


class Conflict(ServiceError):
    status = 409
    code = "conflict"


class Invalid(ServiceError):
    status = 422
    code = "invalid"


class Unauthorised(ServiceError):
    status = 401
    code = "unauthorised"
