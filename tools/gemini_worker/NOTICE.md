# Attribution

`worker.py` is adapted from the shunt plugin in Spotify's
[portal-ai-plugins](https://github.com/spotify/portal-ai-plugins/tree/main/plugins/shunt),
Copyright Spotify AB, licensed under the Apache License 2.0 (`LICENSE-APACHE-2.0` here).

What changed:

- The transport is the Antigravity CLI (`agy`) in headless stream-json mode, not Spotify's
  Portal/AiKA service.
- It is written in standard-library Python only.
- Failures are classified as auth, network or provider, and each call is logged locally with
  sizes only, never file contents.
