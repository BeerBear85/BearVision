# Android App Icon Setup

## Current Status
- App name has been updated from "Drive Videos" to "BearVision"
- AndroidManifest.xml configured to use `@drawable/ic_launcher` as the app icon
- Placeholder icon XML created at `app/src/main/res/drawable/ic_launcher.xml`

## To Complete Icon Setup

To use the BearVision logo as the app icon, you need to copy the logo file:

```bash
# From the android_app directory:
cp ../logo/Logo.png app/src/main/res/drawable/ic_launcher.png
```

Then remove the placeholder XML file:
```bash
rm app/src/main/res/drawable/ic_launcher.xml
```

## For Production

For a production Android app, consider:
1. Creating adaptive icons for Android 8.0+ (API level 26)
2. Providing icons in multiple densities (mdpi, hdpi, xhdpi, xxhdpi, xxxhdpi)
3. Creating rounded and square icon variants

The current logo at `logo/Logo.png` can be used as the base for creating these variants.