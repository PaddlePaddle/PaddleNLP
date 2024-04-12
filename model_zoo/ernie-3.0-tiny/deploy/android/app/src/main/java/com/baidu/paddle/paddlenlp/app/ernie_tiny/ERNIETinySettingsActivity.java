package com.baidu.paddle.paddlenlp.app.ernie_tiny;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.ListPreference;
import android.preference.PreferenceManager;
import android.support.v7.app.ActionBar;

import com.baidu.paddle.paddlenlp.app.R;
import com.baidu.paddle.paddlenlp.ui.view.AppCompatPreferenceActivity;


public class ERNIETinySettingsActivity extends AppCompatPreferenceActivity implements
        SharedPreferences.OnSharedPreferenceChangeListener {
    private static final String TAG = ERNIETinySettingsActivity.class.getSimpleName();
    static public String modelDir = "";
    static public int cpuThreadNum = 2;
    static public String cpuPowerMode = "";
    static public String enableLiteFp16 = "false";
    static public String enableLiteInt8 = "true";

    ListPreference lpChoosePreInstalledModel = null;
    ListPreference lpCPUThreadNum = null;
    ListPreference lpCPUPowerMode = null;
    ListPreference lpEnableLiteFp16 = null;
    ListPreference lpEnableLiteInt8 = null;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.ernie_tiny_settings);
        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }

        // Setup UI components
        lpChoosePreInstalledModel =
                (ListPreference) findPreference(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY));
        lpCPUThreadNum = (ListPreference) findPreference(getString(R.string.CPU_THREAD_NUM_KEY));
        lpCPUPowerMode = (ListPreference) findPreference(getString(R.string.CPU_POWER_MODE_KEY));
        lpEnableLiteFp16 = (ListPreference) findPreference(getString(R.string.ENABLE_LITE_FP16_MODE_KEY));
        lpEnableLiteInt8 = (ListPreference) findPreference(getString(R.string.ENABLE_LITE_INT8_MODE_KEY));
    }

    @SuppressLint("ApplySharedPref")
    private void reloadSettingsAndUpdateUI() {
        SharedPreferences sharedPreferences = getPreferenceScreen().getSharedPreferences();

        String model_dir = sharedPreferences.getString(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY),
                getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_DEFAULT));

        String cpu_thread_num = sharedPreferences.getString(getString(R.string.CPU_THREAD_NUM_KEY),
                getString(R.string.CPU_THREAD_NUM_DEFAULT));
        String cpu_power_mode = sharedPreferences.getString(getString(R.string.CPU_POWER_MODE_KEY),
                getString(R.string.CPU_POWER_MODE_DEFAULT));
        String enable_lite_fp16 = sharedPreferences.getString(getString(R.string.ENABLE_LITE_FP16_MODE_KEY),
                getString(R.string.ENABLE_LITE_FP16_MODE_DEFAULT));
        String enable_lite_int8 = sharedPreferences.getString(getString(R.string.ENABLE_LITE_INT8_MODE_KEY),
                getString(R.string.ENABLE_LITE_FP16_MODE_DEFAULT));

        lpChoosePreInstalledModel.setSummary(model_dir);
        lpChoosePreInstalledModel.setValue(model_dir);
        lpCPUThreadNum.setValue(cpu_thread_num);
        lpCPUThreadNum.setSummary(cpu_thread_num);
        lpCPUPowerMode.setValue(cpu_power_mode);
        lpCPUPowerMode.setSummary(cpu_power_mode);
        lpEnableLiteFp16.setValue(enable_lite_fp16);
        lpEnableLiteFp16.setSummary(enable_lite_fp16);
        lpEnableLiteInt8.setValue(enable_lite_int8);
        lpEnableLiteInt8.setSummary(enable_lite_int8);
    }

    static boolean checkAndUpdateSettings(Context ctx) {
        boolean settingsChanged = false;
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(ctx);

        String model_dir = sharedPreferences.getString(ctx.getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY),
                ctx.getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_DEFAULT));
        settingsChanged |= !modelDir.equalsIgnoreCase(model_dir);
        modelDir = model_dir;

        String cpu_thread_num = sharedPreferences.getString(ctx.getString(R.string.CPU_THREAD_NUM_KEY),
                ctx.getString(R.string.CPU_THREAD_NUM_DEFAULT));
        settingsChanged |= cpuThreadNum != Integer.parseInt(cpu_thread_num);
        cpuThreadNum = Integer.parseInt(cpu_thread_num);

        String cpu_power_mode = sharedPreferences.getString(ctx.getString(R.string.CPU_POWER_MODE_KEY),
                ctx.getString(R.string.CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpuPowerMode.equalsIgnoreCase(cpu_power_mode);
        cpuPowerMode = cpu_power_mode;

        String enable_lite_fp16 = sharedPreferences.getString(ctx.getString(R.string.ENABLE_LITE_FP16_MODE_KEY),
                ctx.getString(R.string.ENABLE_LITE_FP16_MODE_DEFAULT));
        settingsChanged |= !enableLiteFp16.equalsIgnoreCase(enable_lite_fp16);
        enableLiteFp16 = enable_lite_fp16;

        String enable_lite_int8 = sharedPreferences.getString(ctx.getString(R.string.ENABLE_LITE_INT8_MODE_KEY),
                ctx.getString(R.string.ENABLE_LITE_INT8_MODE_DEFAULT));
        settingsChanged |= !enableLiteInt8.equalsIgnoreCase(enable_lite_int8);
        enableLiteInt8 = enable_lite_int8;

        return settingsChanged;
    }

    static void resetSettings() {
        modelDir = "";
        cpuThreadNum = 2;
        cpuPowerMode = "";
        enableLiteFp16 = "false";
        enableLiteInt8 = "true";
    }

    @Override
    protected void onResume() {
        super.onResume();
        getPreferenceScreen().getSharedPreferences().registerOnSharedPreferenceChangeListener(this);
        reloadSettingsAndUpdateUI();
    }

    @Override
    protected void onPause() {
        super.onPause();
        getPreferenceScreen().getSharedPreferences().unregisterOnSharedPreferenceChangeListener(this);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        reloadSettingsAndUpdateUI();
    }
}
