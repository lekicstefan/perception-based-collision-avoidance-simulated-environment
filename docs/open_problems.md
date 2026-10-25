# Open problems

Things we decided to leave for later. We review this list at every stage gate.

## Communication and car
- The build is much slower than the editor. The lidar scan takes 40 ms instead of 4.4 ms. The camera takes about 28 ms instead of 14 ms. We do not know why. We test in the editor. At the end we fix this and measure again in the build.
- The car stays stopped while any message exists. This is the first version. We change it later if needed.
- The odometer is sent in every bundle. You will read the code and decide if it stays.

## Messages
- A message can only move its stop position closer, never further away. We keep this for now.
- A message is removed when its object disappears from view. A person behind a parked car could still step out. We solve this in the risk stage.

## Identification and tracking
- The ground must be detected properly. Far away, the lidar hits the ground in only a few places. We must solve this, not accept it.
- The tracker gives a speed to things that do not move, such as the kerb, the sidewalk and the scenery. This must be fixed.
- Culling means spending less time on objects that cannot reach the car in time. It is decided from limits (maximum speed and acceleration), never from a predicted speed. It is never permanent. It may not be needed, so we measure first.
- Try camera methods alone and together with the lidar.
- Change the lidar position, direction and resolution only as a last resort.
- Measure three things: objects split into pieces, objects seen only in part, and ID switches.

## Process
- Write the README and the design notes after things work.