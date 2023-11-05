#!/usr/bin/bash

USER_ID=${LOCAL_USER_ID:-9001}
GROUP_ID=${LOCAL_GROUP_ID:-$USER_ID}

echo "Starting with UID : $USER_ID, GID: $GROUP_ID"
groupadd -g "$GROUP_ID" usergroup
useradd --shell /bin/bash -u "$USER_ID" -g usergroup -o -c "" -m user
adduser user sudo
export HOME=/home/user

exec gosu user:usergroup "$@"
