$(document).ready(function () {

  $('.loader').hide();
  $('#result').hide();

  $('#query-form-submit-button').click(function () {
      var form_data = new FormData($('#first-query-form')[0]);
      // Show loading animation
      //$(this).hide();
      $('.loader').show();
      $('#result').hide();

      // Make prediction by calling api /predict
      $.ajax({
          type: 'POST',
          url: '/query',
          data: form_data,
          contentType: false,
          cache: false,
          processData: false,
          async: true,
          success: function (graph_url) {
              // Get and display the result
              //var img = document.createElement("IMG");
              document.getElementById("image-result").src = graph_url;
              //img.src = graph_url;
              //document.getElementById('result').appendChild(img);
              $('.loader').hide();
              $('#result').fadeIn(600);
              //$('#result').text(' Result:  ' + data);
              console.log('Success!');
          },
      });
    });

});
