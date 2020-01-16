#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

name = os.environ.get("SOURCE_BRANCH_NAME", "latest")
version = os.environ.get("SOURCE_VERSION", None)

with open("versions.json", "r") as f:
    versions = json.load(f)

versions[name] = version

with open("versions.json", "w") as f:
    json.dump(versions, f, indent=2)
