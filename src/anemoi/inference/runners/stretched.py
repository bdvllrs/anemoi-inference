# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import List

from anemoi.inference.runners.default import DefaultRunner
from anemoi.inference.types import IntArray

from ..forcings import Forcings
from . import runner_registry

LOG = logging.getLogger(__name__)


@runner_registry.register("stretched")
class StretchdRunner(DefaultRunner):
    def create_boundary_forcings(
        self, variables: List[str], mask: IntArray
    ) -> List[Forcings]:
        """
        No boundaries needed for stretched grid
        """
        return []
