# Android Development Setup

To make the Android app work locally and with GitHub Actions, you need to install the Android SDK.

## Quick Setup Steps:

1. **Download Android Studio**: https://developer.android.com/studio
   - Download the .exe installer for Windows
   - Install Android Studio with default settings
   - This will install the Android SDK automatically

2. **Set Environment Variables**:
   After installation, add these to your system environment variables:
   ```
   ANDROID_HOME=C:\Users\[YourUsername]\AppData\Local\Android\Sdk
   PATH=%PATH%;%ANDROID_HOME%\tools;%ANDROID_HOME%\platform-tools
   ```

3. **Install Required SDK Components**:
   Open Android Studio and go to Tools > SDK Manager, install:
   - Android SDK Platform 34 (API Level 34)
   - Android SDK Build Tools 34.0.0
   - Android SDK Platform Tools

4. **Test the Build**:
   ```bash
   cd android_app
   gradlew.bat build
   ```

## Alternative: Command Line Tools Only

If you prefer not to install Android Studio:

1. Download Android Command Line Tools:
   https://dl.google.com/android/repository/commandlinetools-win-13114758_latest.zip

2. Extract to `C:\Android\cmdline-tools\latest\`

3. Set ANDROID_HOME=C:\Android

4. Install SDK components:
   ```bash
   C:\Android\cmdline-tools\latest\bin\sdkmanager "platforms;android-34" "build-tools;34.0.0"
   ```

## Current Status
- Java 21 is installed ✓
- Gradle wrapper is configured ✓
- Android Gradle Plugin updated to 8.5.2 ✓
- Target/Compile SDK updated to 34 ✓
- **Still needed**: Android SDK installation