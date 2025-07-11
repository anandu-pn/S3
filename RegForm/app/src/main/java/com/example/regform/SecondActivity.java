package com.example.regform;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

public class SecondActivity extends AppCompatActivity {
    TextView displayData;
    Button backButton;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_second);

        displayData = findViewById(R.id.displayData);
        backButton = findViewById(R.id.backButton);

        Intent intent= getIntent();

        String name = intent.getStringExtra("name");
        String phone = intent.getStringExtra("phone");
        String gender = intent.getStringExtra("Gender");
        String semester = intent.getStringExtra("semester");
        String languages = intent.getStringExtra("languages");

        String result = "Name: " + name + "\n"
                + "Phone: " + phone + "\n"
                + "Gender: " + gender + "\n"
                + "Semester: " + semester + "\n"
                + "Languages: " + languages;
        displayData.setText(result);

        backButton.setOnClickListener(v -> {
            finish(); // Closes SecondActivity and returns to MainActivity
        });

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;

        });
    }
}