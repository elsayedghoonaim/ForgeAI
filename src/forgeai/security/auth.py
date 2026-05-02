"""RBAC primitives for API key and JWT authentication."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum

import jwt
from rich.console import Console
from rich.table import Table

console = Console()


class Role(str, Enum):
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {"admin", "config", "inference", "models", "monitoring"},
    Role.OPERATOR: {"inference", "models", "monitoring"},
    Role.VIEWER: {"monitoring"},
}


@dataclass
class APIKey:
    """An API key with associated role and metadata."""

    key_id: str
    key_hash: str
    role: Role
    name: str = ""
    created_at: str = ""
    expires_at: str | None = None
    is_active: bool = True


@dataclass
class TokenPayload:
    """Decoded JWT token payload."""

    sub: str
    role: Role
    permissions: set[str] = field(default_factory=set)
    exp: float = 0.0
    iat: float = 0.0


class AuthManager:
    """Manage API keys and JWT tokens with role-based access control."""

    def __init__(
        self,
        secret_key: str = "change-me",
        algorithm: str = "HS256",
        token_expire_minutes: int = 60,
    ) -> None:
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expire_minutes = token_expire_minutes
        self._api_keys: dict[str, APIKey] = {}

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    @staticmethod
    def permissions_for_role(role: Role) -> set[str]:
        return set(ROLE_PERMISSIONS.get(role, set()))

    def register_api_key(self, raw_key: str, name: str, role: Role) -> APIKey:
        """Register an existing API key value for bootstrap or migration use."""

        key_id = secrets.token_hex(8)
        api_key = APIKey(
            key_id=key_id,
            key_hash=self._hash_key(raw_key),
            role=role,
            name=name,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._api_keys[key_id] = api_key
        return api_key

    def create_api_key(self, name: str, role: Role) -> tuple[str, APIKey]:
        """
        Create a new API key.

        Returns: ``(raw_key, key_info)`` where the raw key is shown once.
        """

        raw_key = f"vdt_{secrets.token_urlsafe(32)}"
        api_key = self.register_api_key(raw_key=raw_key, name=name, role=role)
        console.print(f"[green]OK[/green] API key created: [bold]{name}[/bold] (role={role.value})")
        return raw_key, api_key

    def validate_api_key(self, raw_key: str) -> APIKey | None:
        """Validate a raw API key and return its info."""

        key_hash = self._hash_key(raw_key)
        for api_key in self._api_keys.values():
            if api_key.is_active and hmac.compare_digest(api_key.key_hash, key_hash):
                return api_key
        return None

    def create_token(self, key_id: str, role: Role) -> str:
        """Create a JWT token for authenticated access."""

        now = datetime.now(timezone.utc)
        payload = {
            "sub": key_id,
            "role": role.value,
            "permissions": list(self.permissions_for_role(role)),
            "iat": now.timestamp(),
            "exp": (now + timedelta(minutes=self.token_expire_minutes)).timestamp(),
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

    def verify_token(self, token: str) -> TokenPayload | None:
        """Verify and decode a JWT token."""

        try:
            data = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return TokenPayload(
                sub=data["sub"],
                role=Role(data["role"]),
                permissions=set(data.get("permissions", [])),
                exp=data["exp"],
                iat=data.get("iat", 0),
            )
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""

        if key_id in self._api_keys:
            self._api_keys[key_id].is_active = False
            return True
        return False

    def list_keys(self) -> list[APIKey]:
        """List all API keys."""

        return list(self._api_keys.values())

    def print_keys(self) -> None:
        """Display API keys as a table."""

        table = Table(title="API Keys")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Role")
        table.add_column("Active", justify="center")
        table.add_column("Created")

        for key in self._api_keys.values():
            table.add_row(
                key.key_id,
                key.name,
                key.role.value,
                "yes" if key.is_active else "no",
                key.created_at[:10],
            )
        console.print(table)
