# Simulator checks

Manual checks of the vehicle simulation.

Environment: Unity 6000.3.25f1, fixed timestep 0.01 s, time scale 1.
Date: 2026-10-03.

Vehicle parameters used: wheelbase 2.7 m, max acceleration 2.5 m/s^2, max braking 8 m/s^2, actuation delay 0.10 s, jerk limit 80 m/s^3, max steering 35°, steering rate 25°/s.

## Kinematic bicycle model

| Check | Setup | Expected | Measured |
|---|---|---|---|
| Constant radius circle, midpoint integration | 5 m/s, steering 5.7°, radius L/tan δ = 27.05 m; position error against the exact circle at t = 10, 20, 30 s | below 0.01 mm | below 0.01 mm at t = 10, 20, 30 s |
| Constant radius circle, plain Euler integration | 5 m/s, steering 5.7°, midpoint off, t = 10 s | about 40 mm | about 40 mm |
| Hill, elevation and pitch at the rear axle | `hill_straight`, 5 m/s, at s ≈ 50, 100, 150, 200 m | 0.87 m / +3.4°, 4.52 m / +2.4°, 4.00 m / −3.7°, 0.35 m / −2.4° | 0.87 m / +3.4°, 4.52 m / +2.4°, 4.00 m / −3.7°, 0.35 m / −2.4° |

## Steering and path following

| Check | Setup | Expected | Measured |
|---|---|---|---|
| Steering step 20° | at standstill | 99 % after 0.80 s, peak rate 25.0°/s | 0.80 s, 25.0°/s |
| Steering step 50° | at standstill | target clamped to 35.0°, 99 % after 1.39 s | clamped to 35.0°, 1.39 s |
| Pure pursuit on `curves` | cruise 14 m/s, lateral accel limit 3 m/s² | finish 32.4 s, stop at s ≈ 258.4 m, peak speed 13.5 m/s, max cross-track 0.13 m, max lateral accel 3.2 m/s², max steering 8°, max steering rate 7°/s | finish 32.4 s, stop at s ≈ 258.4 m, peak speed 13.5 m/s, max cross-track 0.13 m, max lateral accel 3.2 m/s², max steering 8°, max steering rate 7°/s |

## Collision detection (emergency braking commanded at a given gap, 14 m/s)

Expected impact speed v = √(2a(d_stop − gap)) with d_stop = 14.35 m. The recorded value is the speed at the step where the overlap is first detected.

| Gap at the command | Expected | Measured |
|---|---|---|
| no command | impact at 14.00 m/s | 14.00 m/s |
| 20 m | stops with a final gap of 5.65 m | final gap 5.65 m |
| 12 m | impact about 6.1 m/s | 6.08 m/s |
| 10 m | about 8.3 m/s | 8.32 m/s |
| 7 m | about 10.8 m/s | 10.80 m/s |
| 5 m | about 12.2 m/s | 12.16 m/s |
| 3 m | about 13.4 m/s | 13.44 m/s |

## Stopping table and effective actuation delay

Full emergency braking through the command executor from steady speed, distance measured from the step the command is submitted.

Data: `testing/tools/braking/stopping_table.csv`.
Fit: `testing/tools/braking/stopping_fit.json`, `testing/tools/braking/stopping_fit.png`.

| Speed | Expected distance | Expected time | Measured distance | Measured time |
|---|---|---|---|---|
| 20 km/h | 2.759 m | 0.85 s | 2.759 m | 0.85 s |
| 30 km/h | 5.587 m | 1.20 s | 5.587 m | 1.20 s |
| 40 km/h | 9.379 m | 1.54 s | 9.379 m | 1.54 s |
| 50 km/h | 14.136 m | 1.89 s | 14.136 m | 1.89 s |
| 60 km/h | 19.858 m | 2.24 s | 19.858 m | 2.24 s |
| 70 km/h | 26.544 m | 2.59 s | 26.544 m | 2.59 s |
| 80 km/h | 34.194 m | 2.93 s | 34.194 m | 2.93 s |
| 90 km/h | 42.809 m | 3.28 s | 42.809 m | 3.28 s |
| 100 km/h | 52.389 m | 3.63 s | 52.389 m | 3.63 s |
