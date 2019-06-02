/* global */

const port = 3000

var express = require('express')
var app = express()

var path = require('path')
var bodyParser = require('body-parser')
var request = require('request')

var assets = path.join(__dirname, '/assets')
app.use(express.static(assets))
app.use(bodyParser.urlencoded({ extended: true }))
app.set('view engine', 'ejs')

// use the wildcard as a catch all; remember that ORDER MATTERS and routes are matched top to bottom
app.get('*', function (req, res) {
  res.send('YOU ARE A STAR!')
})

app.listen(port, () => console.log(`Example app listening on port ${port}!`))
