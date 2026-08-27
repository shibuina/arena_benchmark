"""Part 1 - generating the five base scenarios."""

from .occupancy import AGENT_RADIUS, BREATHING_ROOM, Occupancy, world_statics
from .populate import (
    MIN_ROUTE_CLEARANCE,
    PROFILES,
    ROUTE_STOPS,
    TALKERS,
    basis_for,
    build,
    clearance,
    robot_zones,
    route_points,
    work_zones,
)
from .regions import REGION_PROFILES, build_regions, meets_route, pick_endpoints, stream_topology, topology
from .variants import (
    BASE_RECIPES,
    MIN_EPISODE_S,
    ROBOT_SPEED_MPS,
    BaseRecipe,
    robot_for,
    route_seconds,
)
from .zone_screen import Verdict, ref_safe_zones, screen_zones
from .zones import Role, classify, classify_all, zones_by_role

__all__ = [
    "AGENT_RADIUS",
    "BASE_RECIPES",
    "BREATHING_ROOM",
    "BaseRecipe",
    "MIN_EPISODE_S",
    "MIN_ROUTE_CLEARANCE",
    "Occupancy",
    "PROFILES",
    "ROUTE_STOPS",
    "TALKERS",
    "REGION_PROFILES",
    "ROBOT_SPEED_MPS",
    "Role",
    "Verdict",
    "basis_for",
    "build",
    "build_regions",
    "classify",
    "classify_all",
    "clearance",
    "meets_route",
    "pick_endpoints",
    "ref_safe_zones",
    "robot_for",
    "robot_zones",
    "route_points",
    "route_seconds",
    "screen_zones",
    "stream_topology",
    "topology",
    "work_zones",
    "world_statics",
    "zones_by_role",
]
