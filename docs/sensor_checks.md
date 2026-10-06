# Sensor checks

Manual checks of the simulated sensors, done before any experiment. Expected values are computed from the exact ray and projection geometry.
Check scene: ideal ground plane at height 0, vehicle at the origin facing +Z and stationary, LiDAR mounted 1.3 m forward and 1.8 m high, camera mounted 1.8 m forward and 1.3 m high.

Environment: Unity 6000.3.25f1
Fixed timestep 0.01 s
Date: 05.10.2026.

## LiDAR against a wall at a known distance

Wall 20 m ahead of the LiDAR, perpendicular to the viewing direction, 100 scans.

| Check | Config | Expected | Measured |
|---|---|---|---|
| Cells hitting the wall per scan | any | 11618 | 11522 |
| Range of those cells | any | 20.00 to 40.49 m | 20.00 to 40.49 m |
| Returned fraction | `lidar_clean` | 1.0000 | 1.0000 |
| Range error mean, std, max | `lidar_clean` | mean about 0 mm, std 2.9 mm, max below 5.1 mm (1 cm quantisation) | 0.09 mm, 2.86 mm, 5.00 mm |
| Returned fraction | `default` | 0.990 (1 % dropout) | 0.990 |
| Range error std | `default` | 20.2 mm (noise 20 mm plus quantisation) | 20.21 mm |
| Range error mean | `default` | below 0.2 mm in magnitude | 0.01 mm |

## LiDAR field of view

Wall 20 m ahead plus four thin poles 10 m ahead of the LiDAR at azimuth +-59° and +-61°.

| Check | Expected | Measured |
|---|---|---|
| Cells hitting the wall (poles in front) | 11474 | 11474 |
| Rows hit, top row elevation, lowest row hit | 20 rows, +10.00°, -4.00° | 20 rows, +10.00°, -4.00° |
| Columns hit, azimuth range | 600, +59.9° to -59.9° | 600, +59.9° to -59.9° |
| Cells hitting the poles at +-59° | 92 each | 90 each |
| Cells hitting the poles at +-61° | 0 each | 0 each |

## LiDAR resolution (target 0.5 m wide and 1.7 m tall, 50 m ahead on the axis)

| Check | Config | Expected | Measured |
|---|---|---|---|
| Cells hitting the target | `default` (32 beams) | 8 = 4 rows x 2 columns (elevations -0.267, -0.800, -1.333, -1.867°) | 8 |
| Cells hitting the target | `lidar_64` (64 beams) | 16 = 8 rows x 2 columns | 16 |
| Nearest return | any | 49.95 m | 49.95 m |

## Fog

Wall 20 m ahead, 100 scans. Expected fraction = mean over the wall cells of (1 - dropout) * exp(-3.912 r / V).

| Visibility V | Config | Expected returned fraction | Measured |
|---|---|---|---|
| 100 m | `lidar_clean` | 0.384 | 0.3837 |
| 100 m | `default` | 0.380 | 0.3798 |
| 200 m | `lidar_clean` | 0.617 | 0.6165 |
| 200 m | `default` | 0.611 | 0.6102 |
| any | any | cells with no hit stay "max range", never "no return" (sky count unchanged by fog) | sky count with fog off 0, sky count with fog on 0 |

## Camera field of view (640 x 360, horizontal 90°, vertical 58.72°)

Poles and bars 15 m ahead of the camera.

| Object | Expected in the image | Measured |
|---|---|---|
| Red pole at azimuth +-44° | visible at the left and right edges, centres near u = 11 and u = 629 | visible at the left and right edges, centres at u = 11 and u = 629 |
| Blue pole at azimuth +-46° | not visible | not visible |
| Green bar at 28° above the axis | visible at the top edge, near v = 10 | visible at the top edge, at v = 10 |
| Yellow bar at 30° above the axis | not visible | not visible |
| Red and green test cubes | centres at (247, 187) and (393, 187), about 36 px wide, red on the left | centres at (247, 187) and (393, 187), 38 px wide, red on the left |
| Orientation | sky at the top, ground at the bottom, not mirrored | sky at the top, ground at the bottom, not mirrored |

## Noise toggles

| Check | Expected | Measured |
|---|---|---|
| LiDAR noise and dropout off / on | see the wall rows above (std 2.9 mm against 20.2 mm, fraction 1.000 against 0.990) | std 2.86 mm with dropout off against 20.21 mm with dropout on, fraction 1.0000 against 0.990 |
| Camera noise on (std 4), flat region of the image | image standard deviation in the region rises from below 1 level to about 3 to 4 levels | std 0.5 levels with noise off, std 3.4 levels with noise std 4 on |
| Camera exposure gain 0.25 | mean level of a flat region follows the lookup table: 64 -> 30, 128 -> 66, 192 -> 102, 255 -> 137 | 64 -> 30, 128 -> 66, 192 -> 102, 255 -> 137 |
| Camera blur, exposure and noise functions on synthetic data | exposure 12, 30, 66, 102, 137; blur 2 11 39 94 161 216 244 253; noise rms 3.03 and 8.05; row flip PASS | exposure 12, 30, 66, 102, 137; blur 2 11 39 94 161 216 244 253; noise rms 3.03 and 8.05; row flip PASS |
| Pose noise on / off | white noise 0.050 m and 0.100°, bias 0.200 m and 0.200° after 100 s | white noise 0.050 m and 0.100°, bias 0.200 m and 0.200° after 100 s |