"""Item definitions and world-item factories for heal and weapon pickups."""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .entity import ItemEntity


@dataclass(slots=True)
class Item:
    """Base immutable item blueprint used when spawning pickups into the world."""

    item_type: str
    charges: int = 1
    heal_amount: int = 0
    damage_bonus: int = 0
    item_range: int = 0


@dataclass(slots=True)
class MeleeWeaponItem(Item):
    """Weapon pickup that increases melee damage for a limited number of attacks."""

    item_type: str = "melee_weapon"
    charges: int = config.MELEE_WEAPON_ATTACK_CHARGES
    damage_bonus: int = config.MELEE_WEAPON_DAMAGE


@dataclass(slots=True)
class RangedWeaponItem(Item):
    """Weapon pickup that enables directional shots for a limited number of uses."""

    item_type: str = "ranged_weapon"
    charges: int = config.RANGED_WEAPON_SHOT_CHARGES
    damage_bonus: int = config.RANGED_WEAPON_DAMAGE
    item_range: int = config.RANGED_WEAPON_RANGE


@dataclass(slots=True)
class HealItem(Item):
    """Healing pickup that restores health when used."""

    item_type: str = "heal"
    heal_amount: int = config.HEAL_ITEM_AMOUNT
    charges: int = 1


def create_item_entity(entity_id: int, x: int, y: int, item: Item) -> ItemEntity:
    """Convert an immutable item blueprint into a concrete world item entity."""

    return ItemEntity(
        entity_id=entity_id,
        x=x,
        y=y,
        item_type=item.item_type,
        charges=item.charges,
        heal_amount=item.heal_amount,
        damage_bonus=item.damage_bonus,
        item_range=item.item_range,
        reward_lineage_id=entity_id,
    )
