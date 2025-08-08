# Gradle Wrapper JAR Required

The file `gradle-wrapper.jar` is missing and needs to be downloaded.

To fix this, run the following command from the `android_app` directory:

```bash
curl -o gradle/wrapper/gradle-wrapper.jar https://github.com/gradle/gradle/raw/v8.4.0/gradle/wrapper/gradle-wrapper.jar
```

Alternatively, if you have Gradle installed locally, you can regenerate the wrapper:

```bash
cd android_app
gradle wrapper --gradle-version 8.4
```

This will create the missing `gradle-wrapper.jar` file that's needed for the CI to work.