package com.baidu.paddle.paddlenlp.app.ernie_tiny;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Build;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.View;

import com.baidu.paddle.paddlenlp.app.R;

public class ERNIETinyWelcomeActivity extends Activity {
    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.LOLLIPOP) {
            getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                    | View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            );
            getWindow().setStatusBarColor(Color.TRANSPARENT);
        }
        setContentView(R.layout.ernie_tiny_welcome);
    }

    public void startActivity(View view) {
        Intent intent = new Intent(ERNIETinyWelcomeActivity.this, ERNIETinyMainActivity.class);
        startActivity(intent);
    }
}
