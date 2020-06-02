package com.example.syj.centerface;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends Activity
{
    private static final int SELECT_IMAGE = 1;

    private TextView infoResult;
    private ImageView imageView;
    private Bitmap yourSelectedImage = null;

    private centerface squeezencnn = new centerface();

    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        try
        {
            initSqueezeNcnn();
        }
        catch (IOException e)
        {
            Log.e("MainActivity", "initSqueezeNcnn error");
        }

        infoResult = (TextView) findViewById(R.id.infoResult);
        imageView = (ImageView) findViewById(R.id.imageView);

        Button buttonImage = (Button) findViewById(R.id.buttonImage);
        buttonImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                Intent i = new Intent(Intent.ACTION_PICK);
                i.setType("image/*");
                startActivityForResult(i, SELECT_IMAGE);
            }
        });

        Button buttonDetect = (Button) findViewById(R.id.buttonDetect);
        buttonDetect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (yourSelectedImage == null)
                    return;
                long start = System.currentTimeMillis();
                String result = squeezencnn.Detect(yourSelectedImage, false);
                long end = System.currentTimeMillis();
                long time = end - start;
                if (result == null)
                {
                    infoResult.setText("detect failed");
                }
                else
                {
                    infoResult.setText(result+" cpu time: " +time+" ms");
                    imageView.setImageBitmap(yourSelectedImage);
                }
            }
        });

        Button buttonDetectGPU = (Button) findViewById(R.id.buttonDetectGPU);
        buttonDetectGPU.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (yourSelectedImage == null)
                    return;
                long start = System.currentTimeMillis();
                String result = squeezencnn.Detect(yourSelectedImage, true);
                long end = System.currentTimeMillis();
                long time = end - start;
                if (result == null)
                {
                    infoResult.setText("detect failed");
                }
                else
                {
                    infoResult.setText(result+" gpu time: " +time+" ms");
                    imageView.setImageBitmap(yourSelectedImage);
                }
            }
        });
    }

    private void initSqueezeNcnn() throws IOException
    {
        byte[] param = null;
        byte[] bin = null;
//        byte[] words = null;

        {
            InputStream assetsInputStream = getAssets().open("centerface.param.bin");
            int available = assetsInputStream.available();
            param = new byte[available];
            int byteCode = assetsInputStream.read(param);
            assetsInputStream.close();
        }
        {
            InputStream assetsInputStream = getAssets().open("centerface.bin");
            int available = assetsInputStream.available();
            bin = new byte[available];
            int byteCode = assetsInputStream.read(bin);
            assetsInputStream.close();
        }
//        {
//            InputStream assetsInputStream = getAssets().open("synset_words.txt");
//            int available = assetsInputStream.available();
//            words = new byte[available];
//            int byteCode = assetsInputStream.read(words);
//            assetsInputStream.close();
//        }

        squeezencnn.Init(param, bin);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            //https://blog.csdn.net/pbm863521/article/details/73571777 a uri
            // data is a intent
            try
            {
                if (requestCode == SELECT_IMAGE) {
                    Bitmap bitmap = decodeUri(selectedImage);

                    yourSelectedImage = bitmap.copy(Bitmap.Config.ARGB_8888, true);

                    // resize to 227x227
//                    yourSelectedImage = Bitmap.createScaledBitmap(rgba, 300, 300, false);

//                    rgba.recycle();  //回收

                    imageView.setImageBitmap(yourSelectedImage);
                }
            }
            catch (FileNotFoundException e)
            {
                Log.e("MainActivity", "FileNotFoundException");
                return;
            }
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options(); // a class
        o.inJustDecodeBounds = true;  //return size only
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 400;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
                    || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        return BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);
    }

}
