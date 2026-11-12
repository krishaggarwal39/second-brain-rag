"""Operator commands. Usage:

    python -m app.cli promote user@example.com
    python -m app.cli demote user@example.com
    python -m app.cli promote-from-env      # promotes existing users listed in ADMIN_EMAILS

Roles are read from the database on every request, so changes apply immediately.
"""
import os
import sys

from app.core.auth import get_user_store, normalize_email


def set_role_by_email(email: str, role: str) -> bool:
    store = get_user_store()
    user_id = store.get_user_id_by_email(email)
    if user_id is None:
        return False
    store.set_role(user_id, role)
    return True


def promote_admins_from_env() -> list[str]:
    """Promote already-registered users whose email is in ADMIN_EMAILS."""
    raw = os.getenv("ADMIN_EMAILS", "")
    promoted = []
    for email in {normalize_email(e) for e in raw.split(",") if e.strip()}:
        if set_role_by_email(email, "admin"):
            promoted.append(email)
    return sorted(promoted)


def main(argv: list[str]) -> int:
    if len(argv) == 1 and argv[0] == "promote-from-env":
        done = promote_admins_from_env()
        print(f"Promoted {len(done)} user(s): {', '.join(done) or '-'}")
        return 0
    if len(argv) == 2 and argv[0] in ("promote", "demote"):
        role = "admin" if argv[0] == "promote" else "user"
        if set_role_by_email(argv[1], role):
            print(f"{normalize_email(argv[1])} is now {role}")
            return 0
        print("No such user. The account must register first.", file=sys.stderr)
        return 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
