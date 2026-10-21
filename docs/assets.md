# Third-party assets and licences

Every asset not created by the author of this project is listed here with its source and licence. Licence texts that ship with each asset are kept next to its file(s) in the repository.

## Kenney Car Kit

| Item | Value |
|---|---|
| Used for | Visual model of the ego car |
| Author | Kenney (kenney.nl) |
| Source | https://kenney.nl/assets/car-kit |
| Licence | Creative Commons CC0 1.0 Universal |
| Version | 3.1 |
| Downloaded on | 03.10.2026. |
| Location in repository | perception-based-collision-avoidance-simulated-environment\simulation\perception-based-collision-avoidance-simulation\Assets\Asset Packs\KenneyCarKit |

Used for the ego car and, with `Assets/Editor/EnvironmentKit.cs`, for the other vehicles in the scenes (wrapper prefabs `KN_*`).

## Unity Asset Store packs (not in this repository)

The packs below are free on the Unity Asset Store. Their licence is the Unity Asset Store End User Licence Agreement (Standard EULA, extension asset). It allows use in projects and in builds, but not redistribution of the asset files themselves. This repository is public, so **the pack folders are excluded from git** (`.gitignore`). To rebuild the scenes, import the packs again from the Asset Store into `Assets/Asset Packs/<folder>` and run `Tools > Environment Kit > Build kit`; the IDs of the assets stay the same, so scenes and wrapper prefabs find them again.

| Pack | Author | Used for | Folder in `Assets/Asset Packs` | Imported on |
|---|---|---|---|---|
| City Town Lowpoly Pack (Pandazole Ultimate Pack) | Pandazole | buildings, trees, bushes, cones, signs, bins (`PD_*`) | `Pandazole_Ultimate_Pack` | 20.10.2026. |
| Town Pack | SimplePoly | buildings, trees, street lamps, benches, hydrants (`SP_*`) | `SimplePoly - Town Pack` | 20.10.2026. |
| Banana Man | Banana Yellow Games | pedestrian dummy (`Pedestrian_Adult`, `Pedestrian_Child`) | `Banana Yellow Games` | 20.10.2026. |
| Animated Low Poly 3D Models (Animals_FREE) | ithappy | not used so far (its materials need conversion to the Built-in pipeline) | `ithappy` | 20.10.2026. |

Licence terms: https://unity.com/legal/as-terms

## Environment Kit (generated)

| Item | Value |
|---|---|
| Used for | textures (asphalt, sidewalk, curb), materials, wrapper prefabs, showcase scene |
| Author | this project (`Assets/Editor/EnvironmentKit.cs`) |
| Source | textures are drawn by code; the wrapper prefabs refer to the pack models by ID and contain none of their data |
| Licence | part of this repository |
| Location in repository | perception-based-collision-avoidance-simulated-environment\simulation\perception-based-collision-avoidance-simulation\Assets\Environment |

Design and catalogue: `docs/scenery.md`.