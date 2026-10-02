# Thesis notes

## Phase 1
- NetMQ on Unity/Mono (Windows): subscriber connects at TCP level but ZeroMQ
  handshake never completes unless AsyncIO.ForceDotNet.Force() is called on the
  network thread before creating sockets. Diagnosed with the ZMQ socket monitor
  (EVENT_CONNECTED without EVENT_HANDSHAKE_SUCCEEDED).
- Unity settings used: Fixed Timestep 0.01, Run In Background on, VSync off,
  Mono backend, .NET Standard 2.1.
- Use UnityEngine.Debug explicitly (name clash with System.Diagnostics.Debug).