# BearVision Android prototype

The app is a passwordless local-network prototype. A user enters a name, the
email registered in BearVision, and the LAN URL of the BearVision server. The
app can then list and stream only jobs stored under that email.

## Run

1. Install Android Studio with Android SDK 36 and JDK 17 or newer.
2. Open `apps/android` as a project and let Android Studio sync Gradle.
3. Start the BearVision server from `apps/server-control`:

   ```powershell
   node server/server.mjs
   ```

4. Find the server computer's IPv4 address with `ipconfig`.
5. Connect the Android device to the same Wi-Fi and use an address such as
   `http://192.168.1.50:4321` in the app.

The Windows firewall must allow inbound TCP traffic to port `4321`. The admin
interface remains restricted to `127.0.0.1:4320`.

Cleartext HTTP is enabled only for debug builds. This prototype must not be
used with sensitive or production videos because email ownership is not yet
verified.
