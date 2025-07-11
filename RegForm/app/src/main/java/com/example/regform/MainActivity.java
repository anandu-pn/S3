package com.example.regform;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.Switch;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {
    Spinner spinner;

    EditText nameInput,phoneInput;
    RadioGroup genderGroup;
    Spinner semesterSpinner;
    Switch switchPython,switchJava,switchCpp;
    Button submitBtn;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        spinner=findViewById(R.id.spinner);
        ArrayList<String> semesters=new ArrayList<>();
        semesters.add("Sem1");
        semesters.add("Sem2");
        semesters.add("Sem3");
        semesters.add("Sem4");

        ArrayAdapter<String> adapter = new ArrayAdapter<>(
                this,
                android.R.layout.simple_spinner_item,
                semesters
        );
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);

        //HANDLE ITEM SELECTION
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id) {
                String selected= semesters.get(position);
                Toast.makeText(MainActivity.this,"Selected: "+selected,Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent) {

            }
        });
        nameInput = findViewById(R.id.nameInput);
        phoneInput = findViewById(R.id.phoneInput);
        genderGroup = findViewById(R.id.genderGroup);
        //semesterSpinner = findViewById(R.id.spinner);
        switchPython = findViewById(R.id.switchPython);
        switchJava = findViewById(R.id.switchJava);
        switchCpp = findViewById(R.id.switchCpp);
        submitBtn = findViewById(R.id.submitBtn);

        submitBtn.setOnClickListener(v ->{
            String name= nameInput.getText().toString();
            String phone = phoneInput.getText().toString();

            int selectedGenderId= genderGroup.getCheckedRadioButtonId();
            RadioButton genderBtn= findViewById(selectedGenderId);
            String gender= genderBtn !=null ? genderBtn.getText().toString() : "Not selected";

            String semester = spinner.getSelectedItem().toString();

            StringBuilder langs = new StringBuilder();
            if(switchPython.isChecked()) langs.append("Python ");
            if(switchJava.isChecked()) langs.append("Java ");
            if(switchCpp.isChecked()) langs.append("C++ ");

            Intent intent= new Intent(MainActivity.this,SecondActivity.class);
            intent.putExtra("name",name);
            intent.putExtra("phone",phone);
            intent.putExtra("Gender",gender);
            intent.putExtra("semester",semester);
            intent.putExtra("languages", langs.toString().trim());

            startActivity(intent);
        });

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
    }
}
//package com.example.regform;
//
//import android.content.Intent;
//import android.os.Bundle;
//import android.view.View;
//import android.widget.*;
//
//import androidx.appcompat.app.AppCompatActivity;
//
//public class MainActivity extends AppCompatActivity {
//
//    EditText nameInput, phoneInput;
//    RadioGroup genderGroup;
//    Spinner semesterSpinner;
//    Switch switchPython, switchJava, switchCpp;
//    Button submitBtn;
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_main);
//
//        nameInput = findViewById(R.id.editTextText2);
//        phoneInput = findViewById(R.id.editTextText3);
//        genderGroup = findViewById(R.id.radioGroup);
//        semesterSpinner = findViewById(R.id.spinner);
//        switchPython = findViewById(R.id.switch1);
//        switchJava = findViewById(R.id.switch2);
//        switchCpp = findViewById(R.id.switch3);
//        submitBtn = findViewById(R.id.button);
//
//        submitBtn.setOnClickListener(v -> {
//            String name = nameInput.getText().toString();
//            String phone = phoneInput.getText().toString();
//
//            // Gender
//            int selectedGenderId = genderGroup.getCheckedRadioButtonId();
//            RadioButton genderBtn = findViewById(selectedGenderId);
//            String gender = genderBtn != null ? genderBtn.getText().toString() : "Not selected";
//
//            // Semester
//            String semester = semesterSpinner.getSelectedItem().toString();
//
//            // Languages
//            StringBuilder langs = new StringBuilder();
//            if (switchPython.isChecked()) langs.append("Python ");
//            if (switchJava.isChecked()) langs.append("Java ");
//            if (switchCpp.isChecked()) langs.append("C++");
//
//            // Send data to second activity
//            Intent intent = new Intent(MainActivity.this, SecondActivity.class);
//            intent.putExtra("name", name);
//            intent.putExtra("phone", phone);
//            intent.putExtra("gender", gender);
//            intent.putExtra("semester", semester);
//            intent.putExtra("languages", langs.toString().trim());
//
//            startActivity(intent);
//        });
//    }
//}