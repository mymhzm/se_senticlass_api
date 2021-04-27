function getCookie(cname) {
    var name = cname + "=";
    var ca = document.cookie.split(';');
    for(var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
        c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
        return c.substring(name.length, c.length);
        }
    }
    return "";
}

function checkCookie() {
    var username = getCookie("userName");
    if (username != "") {
        $( "#login_btn" ).css( "display", "none" );
        $( "#logout_btn" ).css( "display", "block" );
        $( "#portal_btn" ).css( "display", "block" );
    }else{
        $( "#login_btn" ).css( "display", "block" );
        $( "#logout_btn" ).css( "display", "none" );
        $( "#portal_btn" ).css( "display", "none" );
    }
}

function clearCookie() {
    var cookies = document.cookie.split(";");
    for (var i = 0; i < cookies.length; i++) {
        var cookie = cookies[i];
        var eqPos = cookie.indexOf("=");
        var name = eqPos > -1 ? cookie.substr(0, eqPos) : cookie;
        document.cookie = name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT";
    }
    window.location.replace("/product")
}
