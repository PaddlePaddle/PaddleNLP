package com.baidu.paddle.paddlenlp.app.ernie_tiny;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.EditTextPreference;
import android.preference.ListPreference;
import android.preference.PreferenceManager;
import android.support.v7.app.ActionBar;

import com.baidu.paddle.paddlenlp.app.R;
import com.baidu.paddle.paddlenlp.ui.Utils;
import com.baidu.paddle.paddlenlp.ui.view.AppCompatPreferenceActivity;

import java.util.ArrayList;
import java.util.List;


public class ERNIETinySettingsActivity extends AppCompatPreferenceActivity implements
        SharedPreferences.OnSharedPreferenceChangeListener {
    private static final String TAG = ERNIETinySettingsActivity.class.getSimpleName();
    static public int selectedModelIdx = -1;
    static public String modelDir = "";
    static public int cpuThreadNum = 2;
    static public String cpuPowerMode = "";
    static public String enableLiteFp16 = "true";

    ListPreference lpChoosePreInstalledModel = null;
    EditTextPreference etModelDir = null;
    ListPreference lpCPUThreadNum = null;
    ListPreference lpCPUPowerMode = null;
    ListPreference lpEnableLiteFp16 = null;

    List<String> preInstalledModelDirs = null;
    List<String> preInstalledCPUThreadNums = null;
    List<String> preInstalledCPUPowerModes = null;
    List<String> preInstalledEnableLiteFp16s = null;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.ernie_tiny_settings);
        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }

        // Initialize pre-installed models
        preInstalledModelDirs = new ArrayList<String>();
        preInstalledCPUThreadNums = new ArrayList<String>();
        preInstalledCPUPowerModes = new ArrayList<String>();
        preInstalledEnableLiteFp16s = new ArrayList<String>();
        preInstalledModelDirs.add(getString(R.string.ERNIE_TINY_MODEL_DIR_DEFAULT));
        preInstalledCPUThreadNums.add(getString(R.string.CPU_THREAD_NUM_DEFAULT));
        preInstalledCPUPowerModes.add(getString(R.string.CPU_POWER_MODE_DEFAULT));
        preInstalledEnableLiteFp16s.add(getString(R.string.ENABLE_LITE_FP16_MODE_DEFAULT));

        // Setup UI components
        lpChoosePreInstalledModel =
                (ListPreference) findPreference(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY));
        String[] preInstalledModelNames = new String[preInstalledModelDirs.size()];
        for (int i = 0; i < preInstalledModelDirs.size(); i++) {
            preInstalledModelNames[i] = preInstalledModelDirs.get(i).substring(preInstalledModelDirs.get(i).lastIndexOf("/") + 1);
        }
        lpChoosePreInstalledModel.setEntries(preInstalledModelNames);
        lpChoosePreInstalledModel.setEntryValues(preInstalledModelDirs.toArray(new String[preInstalledModelDirs.size()]));
        lpCPUThreadNum = (ListPreference) findPreference(getString(R.string.CPU_THREAD_NUM_KEY));
        lpCPUPowerMode = (ListPreference) findPreference(getString(R.string.CPU_POWER_MODE_KEY));
        etModelDir = (EditTextPreference) findPreference(getString(R.string.MODEL_DIR_KEY));
        etModelDir.setTitle("Model dir (SDCard: " + Utils.getSDCardDirectory() + ")");
        lpEnableLiteFp16 = (ListPreference) findPreference(getString(R.string.ENABLE_LITE_FP16_MODE_KEY));
    }

    @SuppressLint("ApplySharedPref")
    private void reloadSettingsAndUpdateUI() {
        SharedPreferences sharedPreferences = getPreferenceScreen().getSharedPreferences();

        String selected_model_dir = sharedPreferences.getString(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY),
                getString(R.string.ERNIE_TINY_MODEL_DIR_DEFAULT));
        int selected_model_idx = lpChoosePreInstalledModel.findIndexOfValue(selected_model_dir);
        if (selected_model_idx >= 0 && selected_model_idx < preInstalledModelDirs.size() && selected_model_idx != selectedModelIdx) {
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putString(getString(R.string.MODEL_DIR_KEY), preInstalledModelDirs.get(selected_model_idx));
            editor.putString(getString(R.string.CPU_THREAD_NUM_KEY), preInstalledCPUThreadNums.get(selected_model_idx));
            editor.putString(getString(R.string.CPU_POWER_MODE_KEY), preInstalledCPUPowerModes.get(selected_model_idx));
            editor.putString(getString(R.string.ENABLE_LITE_FP16_MODE_DEFAULT), preInstalledEnableLiteFp16s.get(selected_model_idx));
            editor.commit();
            lpChoosePreInstalledModel.setSummary(selected_model_dir);
            selectedModelIdx = selected_model_idx;
        }

        String model_dir = sharedPreferences.getString(getString(R.string.MODEL_DIR_KEY),
                getString(R.string.ERNIE_TINY_MODEL_DIR_DEFAULT));
        String cpu_thread_num = sharedPreferences.getString(getString(R.string.CPU_THREAD_NUM_KEY),
                getString(R.string.CPU_THREAD_NUM_DEFAULT));
        String cpu_power_mode = sharedPreferences.getString(getString(R.string.CPU_POWER_MODE_KEY),
                getString(R.string.CPU_POWER_MODE_DEFAULT));
        String enable_lite_fp16 = sharedPreferences.getString(getString(R.string.ENABLE_LITE_FP16_MODE_KEY),
                getString(R.string.ENABLE_LITE_FP16_MODE_DEFAULT));

        etModelDir.setSummary(model_dir);
        lpCPUThreadNum.setValue(cpu_thread_num);
        lpCPUThreadNum.setSummary(cpu_thread_num);
        lpCPUPowerMode.setValue(cpu_power_mode);
        lpCPUPowerMode.setSummary(cpu_power_mode);
        lpEnableLiteFp16.setValue(enable_lite_fp16);
        lpEnableLiteFp16.setSummary(enable_lite_fp16);

    }

    static boolean checkAndUpdateSettings(Context ctx) {
        boolean settingsChanged = false;
        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(ctx);

        String model_dir = sharedPreferences.getString(ctx.getString(R.string.MODEL_DIR_KEY),
                ctx.getString(R.string.ERNIE_TINY_MODEL_DIR_DEFAULT));
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

        return settingsChanged;
    }

    static void resetSettings() {
        selectedModelIdx = -1;
        modelDir = "";
        cpuThreadNum = 2;
        cpuPowerMode = "";
        enableLiteFp16 = "true";
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
